// pion-forwarder: Receives RTP packets via UDP and forwards them via WebRTC.
//
// Production-grade Pion WebRTC sender with:
//   - RegisterDefaultInterceptors (NACK responder, SR/RR, TWCC)
//   - Custom RTCP stats interceptor (NACK/PLI/FIR/ReceiverReport logging)
//   - Decoupled UDP reader / WebRTC writer goroutines (channel-based)
//   - Full packet accounting at every boundary
//   - UDP socket buffer sizing (4MB)
//
// HTTP API (port 8090):
//   GET  /health      -> {"ok":true}
//   GET  /sdp/offer   -> {"sdp":"...","type":"offer"}
//   POST /sdp/answer  -> {"sdp":"...","type":"answer"}
//   GET  /state       -> {"ice":"...","connection":"..."}
//   GET  /stats       -> JSON with full packet accounting
//
// UDP listener (port 5004): receives RTP from GStreamer udpsink.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/pion/interceptor"
	"github.com/pion/rtcp"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
)

// ─── Seq Tracker (uint16 wrap-aware) ──────────────────────────────────────────

type seqTracker struct {
	mu          sync.Mutex
	hasFirst    bool
	firstSeq    uint16
	lastSeq     uint16
	totalPkts   uint64
	gapTotal    uint64
	dupTotal    uint64
	lateTotal   uint64
}

func (t *seqTracker) observe(seq uint16) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.hasFirst {
		t.firstSeq = seq
		t.lastSeq = seq
		t.hasFirst = true
		t.totalPkts++
		return
	}

	fwd := seq - t.lastSeq
	if fwd == 0 {
		t.dupTotal++
	} else if fwd > 0 && fwd < 32768 {
		t.gapTotal += uint64(fwd - 1)
		t.lastSeq = seq
	} else {
		t.lateTotal++
	}
	t.totalPkts++
}

func (t *seqTracker) snapshot() map[string]uint64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	return map[string]uint64{
		"total":   t.totalPkts,
		"first":   uint64(t.firstSeq),
		"last":    uint64(t.lastSeq),
		"gaps":    t.gapTotal,
		"dups":    t.dupTotal,
		"late":    t.lateTotal,
	}
}

// ─── Write Stats ──────────────────────────────────────────────────────────────

type writeStats struct {
	attempts  atomic.Uint64
	successes atomic.Uint64
	errors    atomic.Uint64

	mu          sync.Mutex
	errorTypes  map[string]uint64
	lastError   string
	lastErrTime time.Time
}

func newWriteStats() *writeStats {
	return &writeStats{errorTypes: make(map[string]uint64)}
}

func (w *writeStats) recordError(err error) {
	w.errors.Add(1)
	w.mu.Lock()
	s := err.Error()
	if len(s) > 120 {
		s = s[:120]
	}
	w.errorTypes[s]++
	w.lastError = s
	w.lastErrTime = time.Now()
	w.mu.Unlock()
}

func (w *writeStats) snapshot() map[string]interface{} {
	w.mu.Lock()
	ets := make(map[string]uint64, len(w.errorTypes))
	for k, v := range w.errorTypes {
		ets[k] = v
	}
	le := w.lastError
	lt := w.lastErrTime
	w.mu.Unlock()
	return map[string]interface{}{
		"attempts":     w.attempts.Load(),
		"successes":    w.successes.Load(),
		"errors":       w.errors.Load(),
		"error_types":  ets,
		"last_error":   le,
		"last_err_iso": lt,
	}
}

// ─── RTCP Stats + Retransmission Interceptor ──────────────────────────────────

type rtcpStatsInterceptor struct {
	interceptor.NoOp
	nackCount  atomic.Uint64
	pliCount   atomic.Uint64
	firCount   atomic.Uint64
	rrCount    atomic.Uint64

	mu              sync.Mutex
	lastNackSeqs    []uint16
	nackUniqueSeqs  map[uint16]bool
	lastRR          *rtcp.ReceptionReport
	rrLossHistory   []rrSample

	// Retransmission tracking via BindLocalStream
	retxCount       atomic.Uint64
	retxAgeSumMs    atomic.Uint64
	retxAgeMaxMs    atomic.Uint64
	sentSeqsMu      sync.Mutex
	sentSeqs        map[uint16]int64 // seq → send time (unix nano)
}

type rrSample struct {
	ts            time.Time
	fractionLost  uint8
	totalLost     uint32
	jitter        uint32
}

type rtcpStatsFactory struct {
	inst *rtcpStatsInterceptor
}

func newRTCPStatsFactory() *rtcpStatsFactory {
	return &rtcpStatsFactory{inst: &rtcpStatsInterceptor{
		nackUniqueSeqs: make(map[uint16]bool),
		sentSeqs:       make(map[uint16]int64, 2000),
	}}
}

func (f *rtcpStatsFactory) NewInterceptor(_ string) (interceptor.Interceptor, error) {
	return f.inst, nil
}

func (i *rtcpStatsInterceptor) BindRTCPReader(reader interceptor.RTCPReader) interceptor.RTCPReader {
	return interceptor.RTCPReaderFunc(func(buf []byte, attrs interceptor.Attributes) (int, interceptor.Attributes, error) {
		n, a, err := reader.Read(buf, attrs)
		if err != nil || n == 0 {
			return n, a, err
		}

		pkts, parseErr := rtcp.Unmarshal(buf[:n])
		if parseErr != nil {
			return n, a, err
		}

		for _, pkt := range pkts {
			switch p := pkt.(type) {
			case *rtcp.TransportLayerNack:
				var seqs []uint16
				count := 0
				for idx := range p.Nacks {
					p.Nacks[idx].Range(func(seq uint16) bool {
						seqs = append(seqs, seq)
						count++
						return true
					})
				}
				if count > 0 {
					i.nackCount.Add(uint64(count))
					i.mu.Lock()
					for _, s := range seqs {
						i.nackUniqueSeqs[s] = true
					}
					if len(i.lastNackSeqs) < 200 {
						i.lastNackSeqs = append(i.lastNackSeqs, seqs...)
					}
					i.mu.Unlock()
					log.Printf("[rtcp] NACK: %d missing seqs (first=%d last=%d total=%d unique=%d)",
						count, seqs[0], seqs[len(seqs)-1],
						i.nackCount.Load(), i.uniqueNackCount())
				}
			case *rtcp.PictureLossIndication:
				i.pliCount.Add(1)
				log.Printf("[rtcp] PLI received (total=%d) — browser requests keyframe", i.pliCount.Load())
			case *rtcp.FullIntraRequest:
				i.firCount.Add(1)
				log.Printf("[rtcp] FIR received (total=%d) — browser forces keyframe", i.firCount.Load())
			case *rtcp.ReceiverReport:
				i.rrCount.Add(1)
				for idx := range p.Reports {
					rr := p.Reports[idx]
					i.mu.Lock()
					rrCopy := rr
					i.lastRR = &rrCopy
					i.rrLossHistory = append(i.rrLossHistory, rrSample{
						ts: time.Now(), fractionLost: rr.FractionLost,
						totalLost: rr.TotalLost, jitter: rr.Jitter,
					})
					if len(i.rrLossHistory) > 20 {
						i.rrLossHistory = i.rrLossHistory[1:]
					}
					i.mu.Unlock()
					log.Printf("[rtcp] RR: ssrc=%d fraction=%d/256 (%.1f%%) total_lost=%d jitter=%d dlsr=%d",
						rr.SSRC, rr.FractionLost, float64(rr.FractionLost)/256.0*100,
						rr.TotalLost, rr.Jitter, rr.Delay)
				}
			}
		}
		return n, a, err
	})
}

// BindLocalStream wraps the outgoing RTP writer to detect retransmissions.
// When the NACK responder retransmits a packet, it passes through this writer
// with a seq number we've already seen. We count these and measure their age.
func (i *rtcpStatsInterceptor) BindLocalStream(info *interceptor.StreamInfo, writer interceptor.RTPWriter) interceptor.RTPWriter {
	return interceptor.RTPWriterFunc(func(header *rtp.Header, payload []byte, attrs interceptor.Attributes) (int, error) {
		seq := header.SequenceNumber
		nowNs := time.Now().UnixNano()

		i.sentSeqsMu.Lock()
		if prevNs, exists := i.sentSeqs[seq]; exists {
			ageMs := (nowNs - prevNs) / int64(time.Millisecond)
			i.retxCount.Add(1)
			i.retxAgeSumMs.Add(uint64(ageMs))
			for {
				oldMax := i.retxAgeMaxMs.Load()
				if uint64(ageMs) <= oldMax || i.retxAgeMaxMs.CompareAndSwap(oldMax, uint64(ageMs)) {
					break
				}
			}
		}
		i.sentSeqs[seq] = nowNs
		// Periodic cleanup: remove entries older than 5s
		if len(i.sentSeqs) > 3000 {
			cutoff := nowNs - int64(5*time.Second)
			for s, t := range i.sentSeqs {
				if t < cutoff {
					delete(i.sentSeqs, s)
				}
			}
		}
		i.sentSeqsMu.Unlock()

		return writer.Write(header, payload, attrs)
	})
}

func (i *rtcpStatsInterceptor) uniqueNackCount() int {
	i.mu.Lock()
	defer i.mu.Unlock()
	return len(i.nackUniqueSeqs)
}

func (i *rtcpStatsInterceptor) lossRate() float64 {
	i.mu.Lock()
	defer i.mu.Unlock()
	if len(i.rrLossHistory) == 0 {
		return 0
	}
	// Average of last 3 samples
	start := len(i.rrLossHistory)
	if start > 3 {
		start = len(i.rrLossHistory) - 3
	}
	sum := 0.0
	for _, s := range i.rrLossHistory[start:] {
		sum += float64(s.fractionLost) / 256.0
	}
	return sum / float64(len(i.rrLossHistory[start:]))
}

func (i *rtcpStatsInterceptor) snapshot() map[string]interface{} {
	i.mu.Lock()
	var seqs []uint16
	if len(i.lastNackSeqs) > 0 {
		seqs = make([]uint16, len(i.lastNackSeqs))
		copy(seqs, i.lastNackSeqs)
	}
	uniqueNacks := len(i.nackUniqueSeqs)
	var rr map[string]interface{}
	if i.lastRR != nil {
		rr = map[string]interface{}{
			"ssrc":         i.lastRR.SSRC,
			"fraction":     i.lastRR.FractionLost,
			"total_lost":   i.lastRR.TotalLost,
			"jitter":       i.lastRR.Jitter,
			"dlsr":         i.lastRR.Delay,
		}
	}
	rrHist := make([]map[string]interface{}, len(i.rrLossHistory))
	for idx, s := range i.rrLossHistory {
		rrHist[idx] = map[string]interface{}{
			"ts":         s.ts.Format(time.RFC3339Nano),
			"fraction":   s.fractionLost,
			"total_lost": s.totalLost,
			"jitter":     s.jitter,
		}
	}
	i.mu.Unlock()

	retxCount := i.retxCount.Load()
	retxAvgMs := float64(0)
	if retxCount > 0 {
		retxAvgMs = float64(i.retxAgeSumMs.Load()) / float64(retxCount)
	}

	return map[string]interface{}{
		"nack_total":   i.nackCount.Load(),
		"nack_unique":  uniqueNacks,
		"pli":          i.pliCount.Load(),
		"fir":          i.firCount.Load(),
		"rr":           i.rrCount.Load(),
		"nack_seqs":    seqs,
		"last_rr":      rr,
		"rr_history":   rrHist,
		"loss_rate":    i.lossRate(),
		"retx_count":   retxCount,
		"retx_avg_ms":  retxAvgMs,
		"retx_max_ms":  i.retxAgeMaxMs.Load(),
	}
}

// ─── Globals for stats ────────────────────────────────────────────────────────

var (
	udpSeq      = &seqTracker{}
	udpParseErr atomic.Uint64
	udpReadErr  atomic.Uint64
	udpChFull   atomic.Uint64
	wrt         = newWriteStats()
	rtcpInt     *rtcpStatsInterceptor

	stateMu          sync.Mutex
	currentICE       = "new"
	currentConn      = "new"
	selectedCandDesc = ""
	startTime        = time.Now()
)

func envDefault(key, dflt string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return dflt
}

func writeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// ─── Stats snapshot ───────────────────────────────────────────────────────────

func statsSnapshot() map[string]interface{} {
	stateMu.Lock()
	ice, conn, cand := currentICE, currentConn, selectedCandDesc
	stateMu.Unlock()
	return map[string]interface{}{
		"uptime_s":     int(time.Since(startTime).Seconds()),
		"udp":          udpSeq.snapshot(),
		"udp_parse_err": udpParseErr.Load(),
		"udp_read_err":  udpReadErr.Load(),
		"udp_ch_full":   udpChFull.Load(),
		"write":        wrt.snapshot(),
		"rtcp":         rtcpInt.snapshot(),
		"ice":           ice,
		"connection":    conn,
		"candidate":     cand,
	}
}

func logStats() {
	s := statsSnapshot()
	udp := s["udp"].(map[string]uint64)
	w := s["write"].(map[string]interface{})
	r := s["rtcp"].(map[string]interface{})
	log.Printf(
		"[stats] t=%ds | udp: recv=%d seq=[%d..%d] gaps=%d dups=%d late=%d parse_err=%d read_err=%d ch_full=%d "+
			"| wrt: ok=%d err=%d attempt=%d | rtcp: nack=%d(unique=%d) pli=%d fir=%d rr=%d loss=%.1f%% "+
			"| retx: cnt=%d avg=%dms max=%dms | ice=%s conn=%s",
		s["uptime_s"],
		udp["total"], udp["first"], udp["last"], udp["gaps"], udp["dups"], udp["late"],
		s["udp_parse_err"], s["udp_read_err"], s["udp_ch_full"],
		w["successes"], w["errors"], w["attempts"],
		r["nack_total"], r["nack_unique"], r["pli"], r["fir"], r["rr"],
		r["loss_rate"].(float64)*100,
		r["retx_count"], r["retx_avg_ms"], r["retx_max_ms"],
		s["ice"], s["connection"],
	)
}

func dumpFinalStats() {
	log.Printf("[final-stats] dumping JSON")
	data, _ := json.MarshalIndent(statsSnapshot(), "", "  ")
	// Write to AVTR1_LOCAL_STORAGE (persistent Modal volume) if available
	statsDir := os.Getenv("AVTR1_LOCAL_STORAGE")
	if statsDir == "" {
		statsDir = "/tmp"
	}
	statsPath := statsDir + "/pion_stats.json"
	if err := os.WriteFile(statsPath, data, 0644); err != nil {
		log.Printf("[final-stats] write %s error: %v", statsPath, err)
		// Fallback to /tmp
		os.WriteFile("/tmp/pion_stats.json", data, 0644)
	}
	log.Printf("[final-stats] written to %s", statsPath)
	log.Printf("[final-stats] %s", string(data))
}

// congestionSnapshot returns adaptive bitrate guidance for the Python publisher.
func congestionSnapshot() map[string]interface{} {
	lossRate := rtcpInt.lossRate()
	nackTotal := rtcpInt.nackCount.Load()
	uptime := int(time.Since(startTime).Seconds())
	nackRate := float64(0)
	if uptime > 5 {
		nackRate = float64(nackTotal) / float64(uptime)
	}

	congested := lossRate > 0.05 || nackRate > 50

	// Suggest bitrate reduction: if loss > 5%, reduce by 25% steps
	suggested := 2000 // default
	if congested {
		suggested = 1000
	}
	if lossRate > 0.10 {
		suggested = 500
	}

	return map[string]interface{}{
		"congested":         congested,
		"loss_rate":         lossRate,
		"nack_total":        nackTotal,
		"nack_rate_per_s":   nackRate,
		"suggested_bitrate": suggested,
		"uptime_s":          uptime,
	}
}

// ─── Main ─────────────────────────────────────────────────────────────────────

func main() {
	stunServer := envDefault("PION_STUN_SERVER", "stun:stun.cloudflare.com:3478")
	httpPort := envDefault("PION_HTTP_PORT", "8090")
	rtpPort := envDefault("PION_RTP_PORT", "5004")
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.Printf("[pion] starting: stun=%s http=:%s rtp=:%s", stunServer, httpPort, rtpPort)

	// ── Media engine ──
	m := &webrtc.MediaEngine{}
	err := m.RegisterCodec(webrtc.RTPCodecParameters{
		RTPCodecCapability: webrtc.RTPCodecCapability{
			MimeType:    webrtc.MimeTypeH264,
			ClockRate:   90000,
			SDPFmtpLine: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f",
		},
		PayloadType: 96,
	}, webrtc.RTPCodecTypeVideo)
	if err != nil {
		log.Fatalf("[pion] register codec: %v", err)
	}

	// ── Interceptor registry: defaults + custom RTCP stats ──
	ir := &interceptor.Registry{}
	if err := webrtc.RegisterDefaultInterceptors(m, ir); err != nil {
		log.Fatalf("[pion] register default interceptors: %v", err)
	}
	rtcpFac := newRTCPStatsFactory()
	rtcpInt = rtcpFac.inst
	ir.Add(rtcpFac)
	log.Printf("[pion] interceptors: default (NACK responder, SR/RR, TWCC) + RTCP stats")

	api := webrtc.NewAPI(
		webrtc.WithMediaEngine(m),
		webrtc.WithInterceptorRegistry(ir),
	)

	pcConfig := webrtc.Configuration{
		ICEServers: []webrtc.ICEServer{
			{URLs: []string{stunServer}},
		},
		ICETransportPolicy: webrtc.ICETransportPolicyAll,
	}

	pc, err := api.NewPeerConnection(pcConfig)
	if err != nil {
		log.Fatalf("[pion] create PC: %v", err)
	}
	defer pc.Close()

	// ── Create H264 RTP track ──
	track, err := webrtc.NewTrackLocalStaticRTP(
		webrtc.RTPCodecCapability{
			MimeType:    webrtc.MimeTypeH264,
			ClockRate:   90000,
			SDPFmtpLine: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f",
		},
		"avatar-video", "avatar-stream",
	)
	if err != nil {
		log.Fatalf("[pion] create track: %v", err)
	}
	rtpSender, err := pc.AddTrack(track)
	if err != nil {
		log.Fatalf("[pion] add track: %v", err)
	}
	_ = rtpSender

	// ── Connection state monitoring ──
	pc.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		stateMu.Lock()
		currentConn = state.String()
		stateMu.Unlock()
		log.Printf("[pion] connection state: %s", state.String())
		if state == webrtc.PeerConnectionStateFailed || state == webrtc.PeerConnectionStateDisconnected {
			log.Printf("[pion] WARNING: connection %s — dumping stats", state.String())
			logStats()
		}
	})
	pc.OnICEConnectionStateChange(func(state webrtc.ICEConnectionState) {
		stateMu.Lock()
		currentICE = state.String()
		stateMu.Unlock()
		log.Printf("[pion] ICE state: %s", state.String())
	})

	// ── ICE candidate logging ──
	pc.OnICECandidate(func(c *webrtc.ICECandidate) {
		if c == nil {
			log.Printf("[pion] ICE gathering complete")
			return
		}
		log.Printf("[pion] ICE candidate: type=%s proto=%s addr=%s:%d",
			c.Typ, c.Protocol, c.Address, c.Port)
		stateMu.Lock()
		if currentICE == "connected" || currentICE == "completed" {
			selectedCandDesc = fmt.Sprintf("%s/%s:%d", c.Typ, c.Protocol, c.Port)
		}
		stateMu.Unlock()
	})

	// ── Create offer and gather ALL ICE candidates ──
	offer, err := pc.CreateOffer(nil)
	if err != nil {
		log.Fatalf("[pion] create offer: %v", err)
	}
	if err := pc.SetLocalDescription(offer); err != nil {
		log.Fatalf("[pion] set local description: %v", err)
	}

	gatherComplete := webrtc.GatheringCompletePromise(pc)
	log.Printf("[pion] waiting for ICE gathering...")
	select {
	case <-gatherComplete:
		log.Printf("[pion] ICE gathering complete")
	case <-time.After(30 * time.Second):
		log.Printf("[pion] WARNING: ICE gathering timed out after 30s")
	}

	localDesc := pc.LocalDescription()
	if localDesc == nil {
		log.Fatalf("[pion] local description is nil after gathering")
	}
	log.Printf("[pion] offer ready: %d chars", len(localDesc.SDP))

	// ── Answer state ──
	var answerMu sync.Mutex
	answerReceived := false

	// ── HTTP server ──
	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]bool{"ok": true})
	})

	mux.HandleFunc("/state", func(w http.ResponseWriter, r *http.Request) {
		stateMu.Lock()
		writeJSON(w, map[string]string{
			"ice":        currentICE,
			"connection": currentConn,
		})
		stateMu.Unlock()
	})

	mux.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, statsSnapshot())
	})

	mux.HandleFunc("/congestion", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, congestionSnapshot())
	})

	mux.HandleFunc("/sdp/offer", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, sdpMessage{SDP: localDesc.SDP, Type: "offer"})
	})

	mux.HandleFunc("/sdp/answer", func(w http.ResponseWriter, r *http.Request) {
		var msg sdpMessage
		if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
			http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
			return
		}
		answerMu.Lock()
		defer answerMu.Unlock()
		if answerReceived {
			writeJSON(w, map[string]string{"status": "already_set"})
			return
		}
		if err := pc.SetRemoteDescription(webrtc.SessionDescription{
			SDP:  msg.SDP,
			Type: webrtc.SDPTypeAnswer,
		}); err != nil {
			http.Error(w, "set remote desc: "+err.Error(), http.StatusInternalServerError)
			return
		}
		answerReceived = true
		log.Printf("[pion] remote answer set: %d chars", len(msg.SDP))
		writeJSON(w, map[string]string{"status": "ok"})
	})

	go func() {
		log.Printf("[pion] HTTP listening on :%s", httpPort)
		if err := http.ListenAndServe(":"+httpPort, mux); err != nil {
			log.Fatalf("[pion] HTTP server: %v", err)
		}
	}()

	// ── UDP RTP listener ──
	udpAddr, err := net.ResolveUDPAddr("udp", ":"+rtpPort)
	if err != nil {
		log.Fatalf("[pion] resolve udp: %v", err)
	}
	conn, err := net.ListenUDP("udp", udpAddr)
	if err != nil {
		log.Fatalf("[pion] listen udp: %v", err)
	}
	defer conn.Close()

	// ── Increase UDP receive buffer ──
	const rcvBufSize = 4 * 1024 * 1024 // 4 MB
	if err := conn.SetReadBuffer(rcvBufSize); err != nil {
		log.Printf("[pion] WARNING: SetReadBuffer(%d) failed: %v", rcvBufSize, err)
	}
	// Verify actual buffer size
	if sysConn, err := conn.SyscallConn(); err == nil {
		var actualSize int
		sysConn.Control(func(fd uintptr) {
			actualSize, _ = syscall.GetsockoptInt(int(fd), syscall.SOL_SOCKET, syscall.SO_RCVBUF)
		})
		log.Printf("[pion] UDP read buffer: requested=%d actual=%d (doubled by kernel)", rcvBufSize, actualSize)
	}

	log.Printf("[pion] RTP listener on :%s (buffer=%d)", rtpPort, rcvBufSize)

	// ── Decoupled reader/writer via channel ──
	const chSize = 1000
	packetCh := make(chan *rtp.Packet, chSize)
	stopCh := make(chan struct{})

	// Reader goroutine: UDP -> channel
	go func() {
		buf := make([]byte, 1500)
		for {
			select {
			case <-stopCh:
				return
			default:
			}
			n, _, err := conn.ReadFromUDP(buf)
			if err != nil {
				udpReadErr.Add(1)
				continue
			}
			if n < 12 {
				continue
			}
			// Copy buffer — rtp.Packet stores references to raw bytes,
			// and the NACK responder interceptor reads payload via reference.
			pktBuf := make([]byte, n)
			copy(pktBuf, buf[:n])

			pkt := &rtp.Packet{}
			if err := pkt.Unmarshal(pktBuf); err != nil {
				udpParseErr.Add(1)
				continue
			}
			udpSeq.observe(pkt.SequenceNumber)
			select {
			case packetCh <- pkt:
			default:
				udpChFull.Add(1)
			}
		}
	}()

	// Writer goroutine: channel -> WriteRTP
	go func() {
		for pkt := range packetCh {
			wrt.attempts.Add(1)
			if err := track.WriteRTP(pkt); err != nil {
				wrt.recordError(err)
			} else {
				wrt.successes.Add(1)
			}
		}
	}()

	// Periodic stats logger (every 5s)
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				logStats()
			case <-stopCh:
				return
			}
		}
	}()

	// ── Graceful shutdown ──
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)
	<-sigCh
	log.Printf("[pion] shutdown signal received")
	close(stopCh)

	logStats()
	dumpFinalStats()
	time.Sleep(500 * time.Millisecond) // flush
}

type sdpMessage struct {
	SDP  string `json:"sdp"`
	Type string `json:"type"`
}
