/*
 * grayhole-manet.cc: ns-3.40 MANET gray-hole scenario with per-node telemetry.
 * Build: copy to <ns-3-dir>/scratch/ and run ./ns3 build. Run:
 *   ./ns3 run "grayhole-manet --nNodes=50 --mobility=rwp --dropRatio=0.4 \
 *              --grayFrac=0.2 --seed=1 --out=run_rwp_n50_d40_s1.csv"
 * N nodes on a square field, 802.11b ad hoc, AODV, K CBR/UDP flows of 512 B
 * at about 2 pkt/s (nFlows=0 gives K = max(8, N/4)). Mobility: rwp (random
 * waypoint), gm (Gauss-Markov) or rw2d (bounded 2D random walk). Every node
 * runs the same queue disc (DropTail FIFO, 500 packets). Only the selective
 * drop probability differs: 0.0 for benign nodes, dropRatio for gray holes.
 *
 * Counted population P: unicast transit application-data packets seen at the
 * IPv4 routing decision, excluding locally originated, locally delivered,
 * broadcast, multicast and routing-control packets. Each packet is tracked by
 * UID to exactly one outcome at the node: forwarded (dequeued from the node's
 * queue disc toward the device) or dropped (selective, queue overflow, no
 * route, route error, TTL expiry, ARP failure, other IP drop). Events are
 * counted in the window in which they occur (r received, f forwarded, d
 * dropped) and in_flight is the pending count b at each flush, so
 * b(t-1) + r(t) = f(t) + d(t) + b(t) with b(0) = 0. The attack predicate uses
 * source address and packet size while P uses ports, so the code also counts
 * selective drops outside P (expected zero).
 *
 *   data_rx_forward : packets of P at the routing decision
 *   data_forwarded  : packets of P dequeued toward the device
 *   data_dropped    : packets of P dropped, sum over all causes
 *   drop_*          : per-cause drop counters (validation only)
 *   in_flight       : packets of P not yet resolved at the window flush
 *   rreq_forwarded  : AODV RREQs relayed by this node (re-broadcasts only)
 *   rreq_originated : AODV RREQs originated by this node (validation only)
 *   rrep_sent       : AODV RREPs sent by this node
 *   rerr_sent       : AODV RERRs sent by this node (validation only)
 *   data_originated : CBR data packets sourced by this node
 *   avg_neighbors   : mean number of nodes within commRange, sampled at 1 s
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/aodv-module.h"
#include "ns3/applications-module.h"
#include "ns3/traffic-control-module.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("GrayHoleManet");

// Drop causes (per-cause validation counters; the detector sees the aggregate).
enum DropCause
{
  CAUSE_SELECTIVE = 0,   // gray-hole probabilistic drop (attack)
  CAUSE_OVERFLOW,        // queue-disc internal queue full (benign congestion)
  CAUSE_NO_ROUTE,
  CAUSE_ROUTE_ERROR,     // IP: routing protocol error (AODV RERR path)
  CAUSE_TTL_EXPIRED,
  CAUSE_ARP_FAIL,
  CAUSE_IP_OTHER,
  N_CAUSES
};

// Per-node windowed counters (global so trace callbacks can update them).
struct NodeCounters
{
  uint64_t rxForward = 0;   // population P at the routing decision, this window
  uint64_t forwarded = 0;   // P handed off to the device, this window
  uint64_t dropped = 0;     // P dropped (all causes), this window
  uint64_t cause[N_CAUSES] = {0};
  uint64_t rreqRelayed = 0;    // RREQ re-broadcasts (exported: rreq_forwarded)
  uint64_t rreqOriginated = 0; // own RREQs (validation column)
  uint64_t rrepSent = 0;
  uint64_t rerrSent = 0;    // validation only
  uint64_t dataOrig = 0;
  double nbrSum = 0;
  uint32_t nbrObs = 0;
  void Reset() { *this = NodeCounters(); }
};

static std::vector<NodeCounters> g_ctr;          // live, current window
static std::vector<NodeCounters> g_tot;          // whole-run totals (never reset)
static std::vector<std::vector<NodeCounters>> g_windows;
static std::vector<std::vector<uint64_t>> g_inFlightHist;
// Pending transit-data packets per node: UID -> multiplicity (routing loops
// can legitimately revisit a node, so a count, not a set).
static std::vector<std::unordered_map<uint64_t, uint32_t>> g_pending;
static std::vector<uint64_t> g_pendingCount;     // O(1) size incl. multiplicity
static std::vector<Ptr<Ipv4>> g_ipv4;
static std::vector<std::vector<double>> g_adjCount;      // pairwise contact secs
static double g_commRange = 250.0;
static uint16_t g_dataPort = 9000;

static void PendingAdd(uint32_t node, uint64_t uid)
{
  g_pending[node][uid]++;
  g_pendingCount[node]++;
}

// Returns true if uid was pending at this node (and consumes one instance).
static bool PendingConsume(uint32_t node, uint64_t uid)
{
  auto it = g_pending[node].find(uid);
  if (it == g_pending[node].end()) return false;
  if (--(it->second) == 0) g_pending[node].erase(it);
  g_pendingCount[node]--;
  return true;
}

static void RecordIngress(uint32_t node)
{
  g_ctr[node].rxForward++;
  g_tot[node].rxForward++;
}

static void RecordForwarded(uint32_t node)
{
  g_ctr[node].forwarded++;
  g_tot[node].forwarded++;
}

static void RecordDropped(uint32_t node, DropCause c)
{
  g_ctr[node].dropped++;        g_tot[node].dropped++;
  g_ctr[node].cause[c]++;       g_tot[node].cause[c]++;
}

static bool IsLocalAddress(uint32_t node, Ipv4Address addr)
{
  return g_ipv4[node] && g_ipv4[node]->GetInterfaceForAddress(addr) >= 0;
}

static bool IsUnicast(Ipv4Address dst)
{
  return !dst.IsBroadcast() && !dst.IsMulticast()
         && !dst.IsSubnetDirectedBroadcast(Ipv4Mask("255.255.0.0"));
}

// Membership test for population P. payload is the packet without IPv4 header.
static bool InPopulationP(uint32_t node, const Ipv4Header &h,
                          Ptr<const Packet> payload)
{
  if (!IsUnicast(h.GetDestination())) return false;
  if (h.GetProtocol() != 17) return false;              // UDP only
  if (IsLocalAddress(node, h.GetSource())) return false;      // originated here
  if (IsLocalAddress(node, h.GetDestination())) return false; // delivered here
  if (payload->GetSize() < 8) return false;
  UdpHeader udp;
  payload->PeekHeader(udp);
  return udp.GetDestinationPort() == g_dataPort;        // data, not AODV(654)
}

// Selective-drop queue disc, installed on every node (single DropTail FIFO,
// 500 packets). Only the drop probability differs: 0.0 benign, dropRatio gray.
class SelectiveDropQueueDisc : public QueueDisc
{
public:
  static TypeId GetTypeId()
  {
    static TypeId tid = TypeId("SelectiveDropQueueDisc")
      .SetParent<QueueDisc>()
      .SetGroupName("TrafficControl")
      .AddConstructor<SelectiveDropQueueDisc>()
      .AddAttribute("DropProbability",
                    "Probability of dropping a transit data packet "
                    "(0.0 = benign node, >0 = gray-hole).",
                    DoubleValue(0.0),
                    MakeDoubleAccessor(&SelectiveDropQueueDisc::m_dropRatio),
                    MakeDoubleChecker<double>(0.0, 1.0))
      .AddAttribute("SizeThreshold",
                    "Minimum item size (bytes) for the selective drop to "
                    "apply (screens control packets out of the attack).",
                    UintegerValue(200),
                    MakeUintegerAccessor(&SelectiveDropQueueDisc::m_sizeThresh),
                    MakeUintegerChecker<uint32_t>());
    return tid;
  }
  SelectiveDropQueueDisc() : QueueDisc(QueueDiscSizePolicy::SINGLE_INTERNAL_QUEUE) {}

  void Setup(uint32_t nodeId, Ipv4Address own, Ptr<UniformRandomVariable> rng)
  {
    m_nodeId = nodeId; m_own = own; m_rng = rng;
  }
  std::string InternalQueueMaxSize()
  {
    std::ostringstream oss;
    oss << GetInternalQueue(0)->GetMaxSize();
    return oss.str();
  }
  uint64_t m_seen = 0, m_selDropped = 0;   // whole-run attack book-keeping
  // Selective drops of packets outside population P (expected zero).
  uint64_t m_selDroppedOutsideP = 0;

private:
  bool DoEnqueue(Ptr<QueueDiscItem> item) override
  {
    Ptr<Ipv4QueueDiscItem> ip = DynamicCast<Ipv4QueueDiscItem>(item);
    uint64_t uid = item->GetPacket()->GetUid();
    if (ip && m_dropRatio > 0.0)
      {
        const Ipv4Header &h = ip->GetHeader();
        bool transit = (h.GetSource() != m_own);
        if (transit && item->GetSize() >= m_sizeThresh)
          {
            m_seen++;
            if (m_rng->GetValue(0.0, 1.0) < m_dropRatio)
              {
                m_selDropped++;
                // The attack predicate uses source address and size while
                // population P uses ports: count drops outside P.
                if (!InPopulationP(m_nodeId, h, item->GetPacket()))
                  m_selDroppedOutsideP++;
                if (PendingConsume(m_nodeId, uid))
                  RecordDropped(m_nodeId, CAUSE_SELECTIVE);
                DropBeforeEnqueue(item, "grayhole-selective-drop");
                return false;
              }
          }
      }
    bool ok = GetInternalQueue(0)->Enqueue(item);
    if (!ok)
      {
        // FIFO overflow (benign congestion; same limit on every node).
        if (PendingConsume(m_nodeId, uid))
          RecordDropped(m_nodeId, CAUSE_OVERFLOW);
        DropBeforeEnqueue(item, "fifo-overflow");
        return false;
      }
    return ok;
  }
  Ptr<QueueDiscItem> DoDequeue() override
  {
    Ptr<QueueDiscItem> item = GetInternalQueue(0)->Dequeue();
    if (item)
      {
        // Handoff to the device: the node's final forwarded outcome for P.
        uint64_t uid = item->GetPacket()->GetUid();
        if (PendingConsume(m_nodeId, uid))
          RecordForwarded(m_nodeId);
      }
    return item;
  }
  Ptr<const QueueDiscItem> DoPeek() override
  { return GetInternalQueue(0)->Peek(); }
  bool CheckConfig() override
  {
    if (GetNInternalQueues() == 0)
      {
        AddInternalQueue(CreateObjectWithAttributes<DropTailQueue<QueueDiscItem>>(
            "MaxSize", QueueSizeValue(QueueSize("500p"))));
      }
    return true;
  }
  void InitializeParams() override {}

  uint32_t m_nodeId = 0;
  Ipv4Address m_own;
  double m_dropRatio = 0.0;
  uint32_t m_sizeThresh = 200;
  Ptr<UniformRandomVariable> m_rng;
};

NS_OBJECT_ENSURE_REGISTERED(SelectiveDropQueueDisc);

static uint32_t NodeIdFromContext(const std::string &ctx)
{
  // context looks like "/NodeList/12/$ns3::Ipv4L3Protocol/..."
  size_t a = ctx.find("/NodeList/") + 10;
  size_t b = ctx.find('/', a);
  return std::stoul(ctx.substr(a, b - a));
}

// Ingress point of population P: the packet was accepted for forwarding at
// the routing decision (fires after the TTL check, before the egress qdisc).
static void OnUnicastForward(std::string ctx, const Ipv4Header &h,
                             Ptr<const Packet> p, uint32_t)
{
  uint32_t id = NodeIdFromContext(ctx);
  if (!InPopulationP(id, h, p)) return;
  RecordIngress(id);
  PendingAdd(id, p->GetUid());
}

// IP-layer drops of packets in P: (a) refused at the routing decision (no
// route, route error, TTL), where the packet enters P and is dropped at once;
// (b) already accepted (pending) and dropped on the egress path.
static void OnIpv4Drop(std::string ctx, const Ipv4Header &h,
                       Ptr<const Packet> p, Ipv4L3Protocol::DropReason reason,
                       Ptr<Ipv4>, uint32_t)
{
  uint32_t id = NodeIdFromContext(ctx);
  if (!InPopulationP(id, h, p)) return;
  DropCause c;
  switch (reason)
    {
    case Ipv4L3Protocol::DROP_NO_ROUTE:    c = CAUSE_NO_ROUTE;    break;
    case Ipv4L3Protocol::DROP_ROUTE_ERROR: c = CAUSE_ROUTE_ERROR; break;
    case Ipv4L3Protocol::DROP_TTL_EXPIRED: c = CAUSE_TTL_EXPIRED; break;
    default:                               c = CAUSE_IP_OTHER;    break;
    }
  uint64_t uid = p->GetUid();
  if (PendingConsume(id, uid))
    {
      RecordDropped(id, c);              // case (b): post-acceptance drop
    }
  else
    {
      RecordIngress(id);                 // case (a): refused at the decision
      RecordDropped(id, c);
    }
}

// ARP failure toward the next hop: the queued L3 packet is discarded before
// it ever reaches the queue disc. Packet here includes the IPv4 header.
static void OnArpDrop(std::string ctx, Ptr<const Packet> p)
{
  uint32_t id = NodeIdFromContext(ctx);
  Ptr<Packet> q = p->Copy();
  Ipv4Header h;
  if (q->GetSize() < 20) return;
  q->RemoveHeader(h);
  if (!InPopulationP(id, h, q)) return;
  if (PendingConsume(id, p->GetUid()))
    RecordDropped(id, CAUSE_ARP_FAIL);
}

// Counts AODV control (port 654: 1=RREQ, 2=RREP, 3=RERR) and originated data.
static void OnIpv4Tx(std::string ctx, Ptr<const Packet> pkt,
                     Ptr<Ipv4> ipv4, uint32_t)
{
  uint32_t id = NodeIdFromContext(ctx);
  Ptr<Packet> p = pkt->Copy();
  Ipv4Header iph;
  if (p->GetSize() < iph.GetSerializedSize()) return;
  p->RemoveHeader(iph);
  if (iph.GetProtocol() != 17) return;
  UdpHeader udp;
  if (p->GetSize() < 8) return;
  p->RemoveHeader(udp);
  if (udp.GetDestinationPort() == 654 && p->GetSize() >= 1)
    {
      aodv::TypeHeader th;
      p->RemoveHeader(th);
      if (th.IsValid() && th.Get() == aodv::AODVTYPE_RREQ)
        {
          // rreq_forwarded counts relays only: a transmission is a relay if
          // the Originator address is not one of this node's own addresses.
          aodv::RreqHeader rh;
          if (p->GetSize() >= rh.GetSerializedSize())
            {
              p->RemoveHeader(rh);
              if (ipv4->GetInterfaceForAddress(rh.GetOrigin()) >= 0)
                { g_ctr[id].rreqOriginated++; g_tot[id].rreqOriginated++; }
              else
                { g_ctr[id].rreqRelayed++; g_tot[id].rreqRelayed++; }
            }
        }
      else if (th.IsValid() && th.Get() == aodv::AODVTYPE_RREP)
        { g_ctr[id].rrepSent++; g_tot[id].rrepSent++; }
      else if (th.IsValid() && th.Get() == aodv::AODVTYPE_RERR)
        { g_ctr[id].rerrSent++; g_tot[id].rerrSent++; }
    }
  else if (udp.GetDestinationPort() == g_dataPort)
    {
      if (ipv4->GetInterfaceForAddress(iph.GetSource()) >= 0)
        { g_ctr[id].dataOrig++; g_tot[id].dataOrig++; }
    }
}

// Periodic jobs: neighbor sampling (1 s) and window flush (window s).
static void SampleNeighbors(NodeContainer *nodes)
{
  uint32_t n = nodes->GetN();
  std::vector<Vector> pos(n);
  for (uint32_t i = 0; i < n; ++i)
    pos[i] = nodes->Get(i)->GetObject<MobilityModel>()->GetPosition();
  for (uint32_t i = 0; i < n; ++i)
    {
      uint32_t cnt = 0;
      for (uint32_t j = 0; j < n; ++j)
        {
          if (i == j) continue;
          double dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y;
          if (dx * dx + dy * dy <= g_commRange * g_commRange)
            { cnt++; g_adjCount[i][j] += 1.0; }
        }
      g_ctr[i].nbrSum += cnt;
      g_ctr[i].nbrObs += 1;
    }
  Simulator::Schedule(Seconds(1.0), &SampleNeighbors, nodes);
}

static void FlushWindow(uint32_t n)
{
  for (uint32_t i = 0; i < n; ++i)
    {
      g_windows[i].push_back(g_ctr[i]);
      g_inFlightHist[i].push_back(g_pendingCount[i]);
      g_ctr[i].Reset();
    }
}

int main(int argc, char *argv[])
{
  uint32_t nNodes = 50;
  std::string mobility = "rwp";
  double dropRatio = 0.4;
  double grayFrac = 0.2;
  double area = 1000.0;
  double duration = 300.0;
  double window = 20.0;
  uint32_t seed = 1;
  uint32_t nFlows = 0;
  std::string out = "grayhole_run.csv";

  CommandLine cmd(__FILE__);
  cmd.AddValue("nNodes", "number of nodes", nNodes);
  cmd.AddValue("mobility", "rwp|gm|rw2d (grid = deprecated alias for rw2d)",
               mobility);
  cmd.AddValue("dropRatio", "gray-hole data drop probability", dropRatio);
  cmd.AddValue("grayFrac", "fraction of gray-hole nodes", grayFrac);
  cmd.AddValue("area", "square field side (m)", area);
  cmd.AddValue("duration", "sim duration (s)", duration);
  cmd.AddValue("window", "telemetry window (s)", window);
  cmd.AddValue("nFlows", "number of CBR flows (0 = auto: max(8, N/4))", nFlows);
  cmd.AddValue("seed", "RNG run number", seed);
  cmd.AddValue("out", "output CSV path", out);
  cmd.Parse(argc, argv);

  if (mobility == "grid")
    {
      NS_LOG_UNCOND("note: mobility 'grid' is a deprecated alias; the model "
                    "is a bounded 2D random walk (rw2d), not a Manhattan grid");
      mobility = "rw2d";
    }
  if (nFlows == 0)
    nFlows = std::max<uint32_t>(8, nNodes / 4);

  RngSeedManager::SetSeed(12345);
  RngSeedManager::SetRun(seed);

  g_ctr.assign(nNodes, NodeCounters());
  g_tot.assign(nNodes, NodeCounters());
  g_windows.assign(nNodes, {});
  g_inFlightHist.assign(nNodes, {});
  g_pending.assign(nNodes, {});
  g_pendingCount.assign(nNodes, 0);
  g_ipv4.assign(nNodes, nullptr);
  g_adjCount.assign(nNodes, std::vector<double>(nNodes, 0.0));

  NodeContainer nodes;
  nodes.Create(nNodes);

  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211b);
  YansWifiPhyHelper phy;
  YansWifiChannelHelper channel;
  channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
  // Disk-range propagation at commRange, as in the bootstrap simulator.
  channel.AddPropagationLoss("ns3::RangePropagationLossModel",
                             "MaxRange", DoubleValue(g_commRange));
  phy.SetChannel(channel.Create());
  WifiMacHelper mac;
  mac.SetType("ns3::AdhocWifiMac");
  wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                               "DataMode", StringValue("DsssRate11Mbps"),
                               "ControlMode", StringValue("DsssRate1Mbps"));
  NetDeviceContainer devices = wifi.Install(phy, mac, nodes);

  MobilityHelper mob;
  Ptr<UniformRandomVariable> u = CreateObject<UniformRandomVariable>();
  u->SetAttribute("Min", DoubleValue(0.0));
  u->SetAttribute("Max", DoubleValue(area));
  Ptr<RandomRectanglePositionAllocator> posAlloc =
      CreateObject<RandomRectanglePositionAllocator>();
  posAlloc->SetAttribute("X", PointerValue(u));
  posAlloc->SetAttribute("Y", PointerValue(u));
  mob.SetPositionAllocator(posAlloc);

  if (mobility == "rwp") {
    mob.SetMobilityModel("ns3::RandomWaypointMobilityModel",
        "Speed", StringValue("ns3::UniformRandomVariable[Min=1.0|Max=15.0]"),
        "Pause", StringValue("ns3::ConstantRandomVariable[Constant=1.0]"),
        "PositionAllocator", PointerValue(posAlloc));
  } else if (mobility == "gm") {
    mob.SetMobilityModel("ns3::GaussMarkovMobilityModel",
        "Bounds", BoxValue(Box(0, area, 0, area, 0, 100)),
        "TimeStep", TimeValue(Seconds(1.0)),
        "Alpha", DoubleValue(0.75),
        "MeanVelocity", StringValue("ns3::UniformRandomVariable[Min=1.0|Max=15.0]"),
        "MeanDirection", StringValue("ns3::UniformRandomVariable[Min=0|Max=6.283185]"),
        "MeanPitch", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));
  } else if (mobility == "rw2d") {
    // Bounded 2D random walk, time-slotted direction changes (not Manhattan).
    mob.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
        "Bounds", RectangleValue(Rectangle(0, area, 0, area)),
        "Speed", StringValue("ns3::UniformRandomVariable[Min=1.0|Max=15.0]"),
        "Mode", StringValue("Time"),
        "Time", TimeValue(Seconds(2.0)));
  } else {
    NS_ABORT_MSG("unknown mobility '" << mobility << "' (use rwp|gm|rw2d)");
  }
  mob.Install(nodes);

  AodvHelper aodv;
  InternetStackHelper internet;
  internet.SetRoutingHelper(aodv);
  internet.Install(nodes);
  Ipv4AddressHelper addr;
  addr.SetBase("10.1.0.0", "255.255.0.0");
  Ipv4InterfaceContainer ifaces = addr.Assign(devices);
  for (uint32_t i = 0; i < nNodes; ++i)
    g_ipv4[i] = nodes.Get(i)->GetObject<Ipv4>();

  uint32_t nGray = std::max<uint32_t>(1, (uint32_t)std::round(grayFrac * nNodes));
  Ptr<UniformRandomVariable> pick = CreateObject<UniformRandomVariable>();
  Ptr<UniformRandomVariable> dropRng = CreateObject<UniformRandomVariable>();
  std::vector<bool> isGray(nNodes, false);
  std::vector<Ptr<SelectiveDropQueueDisc>> qdiscs(nNodes);
  uint32_t assigned = 0;
  while (assigned < nGray)
    {
      uint32_t idx = (uint32_t)pick->GetInteger(0, nNodes - 1);
      if (isGray[idx]) continue;
      isGray[idx] = true;
      assigned++;
    }

  TrafficControlHelper tchDel;
  for (uint32_t i = 0; i < nNodes; ++i)
    {
      Ptr<NetDevice> dev = devices.Get(i);
      tchDel.Uninstall(dev);                       // remove default root qdisc
      TrafficControlHelper tch;
      tch.SetRootQueueDisc("SelectiveDropQueueDisc",
                           "DropProbability",
                           DoubleValue(isGray[i] ? dropRatio : 0.0));
      QueueDiscContainer qd = tch.Install(dev);
      Ptr<SelectiveDropQueueDisc> g =
          DynamicCast<SelectiveDropQueueDisc>(qd.Get(0));
      g->Setup(i, ifaces.GetAddress(i), dropRng);
      qdiscs[i] = g;
    }

  // Runtime verification of the installed queue discs: query the live objects
  // through the TrafficControlLayer and abort on any mismatch.
  uint32_t rtVerified = 0;
  for (uint32_t i = 0; i < nNodes; ++i)
    {
      Ptr<TrafficControlLayer> tc =
          nodes.Get(i)->GetObject<TrafficControlLayer>();
      NS_ABORT_MSG_IF(!tc, "node " << i << ": no TrafficControlLayer");
      Ptr<QueueDisc> root = tc->GetRootQueueDiscOnDevice(devices.Get(i));
      NS_ABORT_MSG_IF(!root, "node " << i << ": no root queue disc");
      NS_ABORT_MSG_IF(root != qdiscs[i],
                      "node " << i << ": live root disc is not the installed one");
      NS_ABORT_MSG_IF(root->GetInstanceTypeId().GetName()
                          != "SelectiveDropQueueDisc",
                      "node " << i << ": wrong disc type "
                      << root->GetInstanceTypeId().GetName());
      DoubleValue dv;
      root->GetAttribute("DropProbability", dv);
      double want = isGray[i] ? dropRatio : 0.0;
      NS_ABORT_MSG_IF(std::abs(dv.Get() - want) > 1e-12,
                      "node " << i << ": live DropProbability " << dv.Get()
                      << " != " << want);
      // CheckConfig creates the internal queue at initialization: force it now.
      root->Initialize();
      NS_ABORT_MSG_IF(qdiscs[i]->InternalQueueMaxSize() != "500p",
                      "node " << i << ": internal queue "
                      << qdiscs[i]->InternalQueueMaxSize() << " != 500p");
      rtVerified++;
    }
  NS_LOG_UNCOND("runtime queue-disc verification: " << rtVerified << "/"
                << nNodes << " nodes OK");

  // CBR/UDP flows (one sink per unique destination node)
  ApplicationContainer apps;
  std::vector<bool> hasSink(nNodes, false);
  for (uint32_t f = 0; f < nFlows; ++f)
    {
      uint32_t s = pick->GetInteger(0, nNodes - 1);
      uint32_t d = pick->GetInteger(0, nNodes - 1);
      if (s == d) d = (d + 1) % nNodes;
      OnOffHelper onoff("ns3::UdpSocketFactory",
                        InetSocketAddress(ifaces.GetAddress(d), g_dataPort));
      onoff.SetAttribute("DataRate", StringValue("8kbps")); // ~2 pkt/s @512B
      onoff.SetAttribute("PacketSize", UintegerValue(512));
      onoff.SetAttribute("OnTime",
          StringValue("ns3::ConstantRandomVariable[Constant=1.0]"));
      onoff.SetAttribute("OffTime",
          StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));
      apps.Add(onoff.Install(nodes.Get(s)));
      if (!hasSink[d])
        {
          hasSink[d] = true;
          PacketSinkHelper sink("ns3::UdpSocketFactory",
              InetSocketAddress(Ipv4Address::GetAny(), g_dataPort));
          apps.Add(sink.Install(nodes.Get(d)));
        }
    }
  apps.Start(Seconds(2.0));
  apps.Stop(Seconds(duration));

  // Telemetry traces. Queue disc outcomes (selective, overflow, forwarded)
  // are recorded inside SelectiveDropQueueDisc.
  Config::Connect("/NodeList/*/$ns3::Ipv4L3Protocol/UnicastForward",
                  MakeCallback(&OnUnicastForward));
  Config::Connect("/NodeList/*/$ns3::Ipv4L3Protocol/Tx",
                  MakeCallback(&OnIpv4Tx));
  Config::Connect("/NodeList/*/$ns3::Ipv4L3Protocol/Drop",
                  MakeCallback(&OnIpv4Drop));
  // ARP failures discard queued L3 packets before the qdisc. Without this
  // trace such packets stay in in_flight.
  bool arpHooked =
      Config::ConnectFailSafe("/NodeList/*/$ns3::ArpL3Protocol/Drop",
                              MakeCallback(&OnArpDrop));
  if (!arpHooked)
    NS_LOG_UNCOND("note: ArpL3Protocol/Drop trace unavailable; ARP-failed "
                  "packets will appear in in_flight/pending instead");

  Simulator::Schedule(Seconds(1.0), &SampleNeighbors, &nodes);
  for (double t = window; t <= duration + 0.5; t += window)
    Simulator::Schedule(Seconds(t), &FlushWindow, nNodes);

  Simulator::Stop(Seconds(duration + 1.0));
  Simulator::Run();

  // Windowed per-node telemetry (long format). Columns through avg_neighbors
  // feed the detector features; the later columns are for validation.
  std::ofstream ofs(out);
  ofs << "node,is_gray,window,data_rx_forward,data_forwarded,data_dropped,"
         "rreq_forwarded,rrep_sent,data_originated,avg_neighbors,"
         "in_flight,drop_selective,drop_overflow,drop_no_route,"
         "drop_route_error,drop_ttl_expired,drop_arp_fail,drop_ip_other,"
         "rerr_sent,rreq_originated\n";
  for (uint32_t i = 0; i < nNodes; ++i)
    {
      for (size_t w = 0; w < g_windows[i].size(); ++w)
        {
          const NodeCounters &c = g_windows[i][w];
          double nbr = c.nbrObs ? c.nbrSum / c.nbrObs : 0.0;
          ofs << i << "," << (isGray[i] ? 1 : 0) << "," << w << ","
              << c.rxForward << "," << c.forwarded << "," << c.dropped << ","
              << c.rreqRelayed << "," << c.rrepSent << ","
              << c.dataOrig << "," << nbr << ","
              << g_inFlightHist[i][w] << ","
              << c.cause[CAUSE_SELECTIVE] << "," << c.cause[CAUSE_OVERFLOW] << ","
              << c.cause[CAUSE_NO_ROUTE] << "," << c.cause[CAUSE_ROUTE_ERROR] << ","
              << c.cause[CAUSE_TTL_EXPIRED] << "," << c.cause[CAUSE_ARP_FAIL] << ","
              << c.cause[CAUSE_IP_OTHER] << "," << c.rerrSent << ","
              << c.rreqOriginated << "\n";
        }
    }
  ofs.close();

  // Adjacency (fraction of sampled seconds within range)
  std::ofstream afs(out + ".adj.csv");
  double obs = duration;  // ~1 sample/s
  for (uint32_t i = 0; i < nNodes; ++i)
    {
      for (uint32_t j = 0; j < nNodes; ++j)
        afs << (g_adjCount[i][j] / obs) << (j + 1 < nNodes ? "," : "\n");
    }
  afs.close();

  // Run metadata. Per node, rx == forwarded + dropped + pending_end (exact).
  uint64_t selOutsidePTotal = 0;
  for (uint32_t i = 0; i < nNodes; ++i)
    if (qdiscs[i]) selOutsidePTotal += qdiscs[i]->m_selDroppedOutsideP;
  std::ofstream mfs(out + ".meta.json");
  mfs << "{\n"
      << "  \"schemaRev\": 3,\n"
      << "  \"scenario\": {\n"
      << "    \"nNodes\": " << nNodes << ",\n"
      << "    \"mobility\": \"" << mobility << "\",\n"
      << "    \"mobilityModel\": \""
      << (mobility == "rwp" ? "RandomWaypointMobilityModel"
          : mobility == "gm" ? "GaussMarkovMobilityModel"
                             : "RandomWalk2dMobilityModel") << "\",\n"
      << "    \"dropRatio\": " << dropRatio << ",\n"
      << "    \"grayFrac\": " << grayFrac << ",\n"
      << "    \"nGray\": " << nGray << ",\n"
      << "    \"area_m\": " << area << ",\n"
      << "    \"duration_s\": " << duration << ",\n"
      << "    \"window_s\": " << window << ",\n"
      << "    \"nFlows\": " << nFlows << ",\n"
      << "    \"flowRule\": \"max(8, N/4)\",\n"
      << "    \"pktSizeBytes\": 512,\n"
      << "    \"dataRate\": \"8kbps\",\n"
      << "    \"dataPort\": " << g_dataPort << ",\n"
      << "    \"commRange_m\": " << g_commRange << ",\n"
      << "    \"wifiStandard\": \"80211b\",\n"
      << "    \"routing\": \"AODV\",\n"
      << "    \"seed\": " << seed << "\n"
      << "  },\n"
      << "  \"queueDisc\": {\n"
      << "    \"type\": \"SelectiveDropQueueDisc\",\n"
      << "    \"internalQueue\": \"DropTailQueue\",\n"
      << "    \"maxSize\": \"500p\",\n"
      << "    \"sizeThreshold\": 200,\n"
      << "    \"installedOnAllNodes\": true,\n"
      << "    \"benignDropProbability\": 0.0,\n"
      << "    \"grayDropProbability\": " << dropRatio << ",\n"
      << "    \"runtimeVerifiedNodes\": " << rtVerified << ",\n"
      << "    \"runtimeVerification\": \"live TrafficControlLayer query: "
         "root disc identity, type, DropProbability, internal 500p\",\n"
      << "    \"selectiveDropsOutsideP\": " << selOutsidePTotal << "\n"
      << "  },\n"
      << "  \"nodes\": [\n";
  for (uint32_t i = 0; i < nNodes; ++i)
    {
      const NodeCounters &t = g_tot[i];
      double prob = 0.0;
      if (qdiscs[i])
        {
          DoubleValue dv;
          qdiscs[i]->GetAttribute("DropProbability", dv);
          prob = dv.Get();
        }
      mfs << "    {\"id\": " << i << ", \"isGray\": " << (isGray[i] ? 1 : 0)
          << ", \"dropProbability\": " << prob
          << ", \"rx\": " << t.rxForward
          << ", \"forwarded\": " << t.forwarded
          << ", \"dropped\": " << t.dropped
          << ", \"pending_end\": " << g_pendingCount[i]
          << ", \"rreqRelayed\": " << t.rreqRelayed
          << ", \"rreqOriginated\": " << t.rreqOriginated
          << ", \"selOutsideP\": "
          << (qdiscs[i] ? qdiscs[i]->m_selDroppedOutsideP : 0)
          << ", \"causes\": {"
          << "\"selective\": " << t.cause[CAUSE_SELECTIVE]
          << ", \"overflow\": " << t.cause[CAUSE_OVERFLOW]
          << ", \"no_route\": " << t.cause[CAUSE_NO_ROUTE]
          << ", \"route_error\": " << t.cause[CAUSE_ROUTE_ERROR]
          << ", \"ttl_expired\": " << t.cause[CAUSE_TTL_EXPIRED]
          << ", \"arp_fail\": " << t.cause[CAUSE_ARP_FAIL]
          << ", \"ip_other\": " << t.cause[CAUSE_IP_OTHER] << "}}"
          << (i + 1 < nNodes ? ",\n" : "\n");
    }
  mfs << "  ]\n}\n";
  mfs.close();

  uint64_t seen = 0, drp = 0;
  for (uint32_t i = 0; i < nNodes; ++i)
    if (isGray[i] && qdiscs[i])
      { seen += qdiscs[i]->m_seen; drp += qdiscs[i]->m_selDropped; }
  NS_LOG_UNCOND("wrote " << out << "  gray=" << nGray
                << "  flows=" << nFlows
                << "  transit_seen=" << seen << "  gray_dropped=" << drp
                << "  sel_outside_P=" << selOutsidePTotal
                << " (must be 0)");

  Simulator::Destroy();
  return 0;
}
