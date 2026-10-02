"""Systems challenge pack for the self-improvement arena.

Concrete, verifiable engineering challenges across the systems
spectrum: systems design, distributed systems, concurrency,
databases, networking, compilers, runtimes, and packaging.
Every entry is a task with a checkable acceptance criterion —
no essay prompts.
"""

from __future__ import annotations

from ..challenges import challenge
from ..topics import register_topic_pack

_PACK = {
    "systems_design": [
        challenge(
            "Build a capacity-planning calculator that, given target RPS, "
            "average response size, and a redundancy factor, outputs servers "
            "needed, peak bandwidth, and monthly transfer",
            1,
            "CLI prints correct server count and bandwidth for a worked fixture "
            "(100k RPS, 2KB avg, 3x redundancy) matching hand-computed values within 1%",
            "build", "capacity-planning", "sizing"),
        challenge(
            "Implement a consistent-hashing ring in pure Python with 128 "
            "virtual nodes per physical node and measure the key distribution "
            "for 10k keys across 8 nodes",
            1,
            "script reports max/min node load ratio < 1.3 and removing one node "
            "moves < 1/8 of keys ±2%",
            "code", "sharding", "consistent-hashing"),
        challenge(
            "Write an SLO math tool that converts availability targets "
            "(99.9%, 99.99%) into allowed error budgets per 30-day window and "
            "prints burn-rate alert thresholds",
            1,
            "output matches 43.8m/month budget for 99.9% and 4.38m for 99.99% "
            "to within 1s, plus 2x and 14.4x burn-rate thresholds",
            "code", "slo", "reliability"),
        challenge(
            "Implement a token-bucket and a leaky-bucket rate limiter pair with "
            "a benchmark that proves burst absorption vs smoothing over a 60s "
            "synthetic load trace",
            2,
            "pytest passes: token bucket allows a 100-request burst instantly, "
            "leaky bucket spreads it over ≥10s; report prints both denial curves",
            "code", "rate-limiting", "backpressure"),
        challenge(
            "Build a sharding simulator that compares range, hash, and "
            "directory-based sharding on a skewed Zipf workload and measures "
            "rebalance cost when adding a node",
            2,
            "benchmark reports hot-shard load skew and rebalance moved-key counts "
            "for all 3 schemes on 1M keys",
            "code", "sharding", "simulation"),
        challenge(
            "Implement a circuit-breaker plus hedged-requests client over a "
            "simulated flaky upstream and measure p99 latency vs success rate "
            "at 30% upstream failure",
            2,
            "pytest passes: p99 < 2x baseline with hedging enabled; circuit opens "
            "within 5s of sustained 50% failure",
            "code", "resilience", "circuit-breaker"),
        challenge(
            "Write a backpressure benchmark: unbounded queue vs bounded queue "
            "with load shedding vs CoDel-style dropping on a producer/consumer pair",
            2,
            "script reports consumer p99 latency and dropped/accepted counts for "
            "all three strategies under 2x overload",
            "code", "backpressure", "queues"),
        challenge(
            "Implement a quorum calculator for N=5 nodes showing read/write "
            "quorum sizes under different consistency choices, with a chaos run "
            "proving lost updates at W=1",
            2,
            "pytest passes: calculator outputs R+W>N intersections; chaos run "
            "demonstrates ≥1 lost update with W=1 over 500 writes",
            "code", "quorums", "consistency"),
        challenge(
            "Design and implement a Raft-based config store for 3 nodes over "
            "in-memory queues, then kill the leader mid-write and prove the "
            "write survives with exactly-once client semantics",
            3,
            "pytest passes: 50 leader-kill runs, write committed on majority and "
            "applied exactly once each run",
            "code", "raft", "consensus"),
        challenge(
            "Build a multi-region write simulator that quantifies the "
            "latency/consistency trade-off of sync vs async replication across "
            "3 simulated regions with a partition injector",
            3,
            "simulation report shows p99 write latency and stale-read probability "
            "for sync, async, and quorum modes under 3 partition scenarios",
            "research", "replication", "trade-offs"),
        challenge(
            "Implement a distributed rate limiter using the GCRA algorithm over "
            "a simulated Redis cluster with clock skew, measuring fairness "
            "across 10 clients",
            3,
            "pytest passes: 10 clients each get ≥90% of fair share over 10k "
            "requests with 50ms injected clock skew",
            "code", "rate-limiting", "distributed"),
        challenge(
            "Design a snowflake-style ID generator cluster: implement 4 "
            "generators with epoch handling and prove global uniqueness plus "
            "time-ordering across 1M generated IDs",
            3,
            "script proves 1M IDs unique, monotonic per worker, and k-ordered "
            "within 10ms tolerance",
            "code", "id-generation", "distributed"),
    ],
    "distributed": [
        challenge(
            "Implement vector clocks for 3 processes and write a merge function "
            "that detects concurrent vs causally-ordered events",
            1,
            "pytest passes: happens-before, concurrent, and merge cases all "
            "classified correctly over a scripted event log",
            "code", "vector-clocks", "causality"),
        challenge(
            "Implement a SWIM-style failure detector simulation with 5 nodes "
            "and measure detection time vs false-positive rate under 10% "
            "packet loss",
            1,
            "simulation reports median detection time and zero false suspicions "
            "in 100 no-failure rounds",
            "code", "failure-detection", "gossip"),
        challenge(
            "Build a gossip protocol simulator where 50 nodes disseminate a "
            "rumor and measure rounds-to-convergence with fanout 3",
            1,
            "script reports all 50 nodes infected within 12 rounds median "
            "across 200 runs",
            "code", "gossip", "epidemic"),
        challenge(
            "Implement Raft leader election for a 3-node cluster over "
            "in-memory message queues",
            2,
            "pytest passes: 3 nodes elect exactly one leader within 2s, "
            "50 randomized runs",
            "code", "raft", "leader-election"),
        challenge(
            "Implement a G-counter and a PN-counter CRDT with merge, and prove "
            "convergence after random partitions",
            2,
            "pytest passes: 200 random merge/partition schedules converge to "
            "identical state on all replicas",
            "code", "crdt", "convergence"),
        challenge(
            "Implement single-decree Paxos for 3 acceptors and demonstrate it "
            "survives one acceptor failure",
            2,
            "pytest passes: value chosen and learned with 1 of 3 acceptors down, "
            "30 runs",
            "code", "paxos", "consensus"),
        challenge(
            "Implement a distributed lock with fencing tokens over a simulated "
            "lock server and prove a stale holder cannot write after lease expiry",
            2,
            "pytest passes: stale writer's write rejected by fencing token in "
            "100% of 200 simulated races",
            "code", "locking", "fencing"),
        challenge(
            "Build a consistent-hashing membership service with join/leave and "
            "a failure injector that verifies no data loss during 20 churn events",
            2,
            "pytest passes: 10k keys survive 20 random join/leave/fail events "
            "with zero loss",
            "code", "membership", "churn"),
        challenge(
            "Implement full Raft log replication with snapshotting for a "
            "5-node cluster and pass a linearizability checker on a register "
            "workload",
            3,
            "linearizability checker passes on a 5k-operation history with "
            "random partitions and leader kills",
            "code", "raft", "linearizability"),
        challenge(
            "Implement an LWW-element-set and an OR-set CRDT pair, "
            "fuzz-compare their semantics, and document exactly where they diverge",
            3,
            "fuzzer runs 10k random op histories; divergence report lists the "
            "exact add/remove interleavings where LWW-set and OR-set disagree",
            "research", "crdt", "semantics"),
        challenge(
            "Build a byzantine-tolerant broadcast (Bracha reliable broadcast) "
            "for 4 nodes with 1 byzantine node and prove agreement despite "
            "equivocation",
            3,
            "pytest passes: all 3 honest nodes deliver identical value across "
            "100 runs with an adversarial byzantine node",
            "code", "byzantine", "broadcast"),
        challenge(
            "Implement a distributed transaction coordinator using 2PC with a "
            "participant-failure injector and measure blocking windows",
            3,
            "simulation reports coordinator blocking-time distribution and proves "
            "no split commit across 500 transactions with failures",
            "code", "2pc", "transactions"),
    ],
    "concurrency": [
        challenge(
            "Implement a bounded blocking queue with condition variables and "
            "prove FIFO ordering under 4 producers",
            1,
            "pytest passes: 10k items dequeued in exact FIFO order with 4 "
            "producers and 2 consumers",
            "code", "queues", "threads"),
        challenge(
            "Build a thread-pool executor from scratch (submit/future/shutdown) "
            "and benchmark throughput vs spawning a thread per task",
            1,
            "benchmark reports pool ≥5x faster than thread-per-task for 10k "
            "no-op tasks",
            "code", "thread-pool", "benchmark"),
        challenge(
            "Implement the dining-philosophers problem with a deadlock-free "
            "resource-ordering solution and run it for 60s",
            1,
            "script runs 60s with 5 philosophers, zero deadlocks, each eats "
            "≥10 times",
            "code", "deadlock", "classic"),
        challenge(
            "Implement a lock-free Treiber stack with CAS-style semantics and "
            "stress-test it with 8 threads",
            2,
            "pytest passes: 100k pushes/pops across 8 threads, final size exact, "
            "ABA scenario covered",
            "code", "lock-free", "stack"),
        challenge(
            "Build an actor-model runtime (mailboxes, spawn, link/supervise) "
            "and implement a supervised worker pool that restarts crashed workers",
            2,
            "pytest passes: killing 5 of 10 workers mid-job still completes all "
            "1k jobs",
            "code", "actors", "supervision"),
        challenge(
            "Implement a readers-writer lock with writer preference and prove "
            "no writer starvation under continuous readers",
            2,
            "pytest passes: writer acquires within 100ms while 16 readers "
            "hammer, 50 trials",
            "code", "locks", "starvation"),
        challenge(
            "Write a deadlock detector that builds a wait-for graph from "
            "lock-acquire events and flags cycles in a trace",
            2,
            "detector flags all 3 planted deadlocks in a synthetic trace and "
            "reports zero false positives on 5 clean traces",
            "code", "deadlock", "detection"),
        challenge(
            "Implement a work-stealing deque and scheduler, and show speedup on "
            "a recursive parallel quicksort",
            2,
            "benchmark reports ≥2.5x speedup on 4 cores for parallel quicksort "
            "of 1M ints vs sequential",
            "code", "work-stealing", "scheduling"),
        challenge(
            "Implement a lock-free MPMC ring buffer with sequence numbers "
            "(Disruptor-style) and prove no loss under 4 producers x 4 consumers",
            3,
            "pytest passes: 4 producers x 4 consumers, 100k items, no loss, "
            "no duplicates",
            "code", "lock-free", "ring-buffer"),
        challenge(
            "Build a software transactional memory prototype (optimistic, "
            "versioned objects) and run a bank-transfer benchmark",
            3,
            "pytest passes: 50k transfers across 8 threads keep total balance "
            "invariant, abort rate < 20%",
            "code", "stm", "transactions"),
        challenge(
            "Implement a hazard-pointer memory reclamation scheme for a "
            "lock-free linked list and prove no use-after-free under stress",
            3,
            "stress test runs 30s with 8 threads doing inserts/deletes; "
            "accounting shows zero dangling dereferences",
            "code", "lock-free", "memory-reclamation"),
        challenge(
            "Build a deterministic concurrency testing harness (controlled "
            "scheduler with preemption points) and use it to find a race in a "
            "sample program",
            3,
            "harness finds the planted race within 1k schedules and the fixed "
            "program passes 10k schedules",
            "build", "testing", "determinism"),
    ],
    "databases": [
        challenge(
            "Implement a B-tree (order 4) with insert/search/delete and verify "
            "against a sorted-list oracle",
            1,
            "pytest passes: 5k random ops match the sorted-list oracle exactly",
            "code", "b-tree", "indexing"),
        challenge(
            "Build a write-ahead log with checkpointing and crash recovery for "
            "a toy KV store",
            1,
            "pytest passes: 200 random crash points recover to the last "
            "committed state",
            "code", "wal", "recovery"),
        challenge(
            "Implement a cost-based index selector that picks between full "
            "scan and B-tree index given table stats",
            1,
            "selector chooses index for selectivity < 5% and scan above, "
            "matching hand-computed break-even on 10 fixtures",
            "code", "query-planning", "indexes"),
        challenge(
            "Implement an LSM-tree with memtable, SSTables, and leveled "
            "compaction, and benchmark write throughput vs the B-tree",
            2,
            "benchmark reports LSM ≥3x write throughput of B-tree on 200k "
            "inserts; reads verified correct via oracle",
            "code", "lsm", "storage-engine"),
        challenge(
            "Implement MVCC with snapshot isolation for a toy row store and "
            "demonstrate a write-skew anomaly",
            2,
            "pytest passes: concurrent transactions see consistent snapshots; "
            "write-skew demo produces the documented anomaly",
            "code", "mvcc", "isolation"),
        challenge(
            "Build a simple SQL query planner (filter pushdown, join ordering "
            "by cardinality) for a 3-table schema",
            2,
            "planner emits optimal join order on 5 fixtures; EXPLAIN-style output "
            "shows pushdown applied",
            "code", "query-planning", "optimizer"),
        challenge(
            "Implement ARIES-style recovery (analysis/redo/undo) on a "
            "page-based store and survive a crash mid-checkpoint",
            2,
            "pytest passes: recovery after crash at 10 random points yields "
            "pages identical to a no-crash run",
            "code", "aries", "recovery"),
        challenge(
            "Build a columnar storage prototype with RLE and dictionary "
            "encoding and measure compression vs a row store",
            2,
            "script reports ≥4x compression on a skewed 1M-row fixture with "
            "correct scan results",
            "code", "columnar", "compression"),
        challenge(
            "Implement a B+tree with concurrent latch crabbing and prove "
            "serializable inserts under 8 threads",
            3,
            "pytest passes: 50k concurrent inserts, tree validates "
            "(ordering + occupancy), zero lost keys",
            "code", "b-tree", "concurrency"),
        challenge(
            "Build a distributed 2PC transaction layer over two toy KV nodes "
            "and prove atomicity under coordinator crash",
            3,
            "pytest passes: 300 transactions with random coordinator crashes — "
            "all commit or all abort, never partial",
            "code", "2pc", "distributed"),
        challenge(
            "Implement a vectorized query execution engine (selection + hash "
            "join on column batches) and benchmark vs row-at-a-time",
            3,
            "benchmark reports ≥5x speedup on a 1M-row filter+join query with "
            "identical results",
            "code", "vectorized", "execution"),
        challenge(
            "Implement a learned-index prototype (RMI over sorted keys) and "
            "compare lookup latency vs B-tree on 10M keys",
            3,
            "script reports RMI mean lookup latency and error-bound correction "
            "cost vs B-tree on 10M sorted keys",
            "research", "learned-index", "indexing"),
    ],
    "networking": [
        challenge(
            "Implement a DNS resolver that parses real DNS wire-format "
            "responses for A/AAAA records from a captured packet file",
            1,
            "script decodes 10 captured responses (incl. compression pointers) "
            "matching dig output",
            "build", "dns", "parsing"),
        challenge(
            "Build a TCP state-machine simulator that walks through handshake, "
            "data transfer, and FIN teardown",
            1,
            "pytest passes: all 11 TCP states reachable; invalid transitions "
            "rejected",
            "code", "tcp", "state-machine"),
        challenge(
            "Implement a token-bucket traffic shaper and verify it caps a "
            "100Mbps flow at 10Mbps over 30s",
            1,
            "simulation shows average ≤10.5Mbps and burst ≤ configured bucket size",
            "code", "traffic-shaping", "qos"),
        challenge(
            "Implement a QUIC-style packet number decoder and stream "
            "multiplexer over UDP loopback",
            2,
            "pytest passes: 3 streams interleave on one UDP socket, reassembled "
            "in order, 10k packets",
            "code", "quic", "multiplexing"),
        challenge(
            "Build an HTTP load balancer (round-robin, least-connections, "
            "consistent-hash) over 4 backend processes",
            2,
            "benchmark reports request distribution within 5% of ideal for "
            "round-robin and key stickiness for consistent-hash",
            "build", "load-balancing", "http"),
        challenge(
            "Implement a congestion-control simulator comparing Reno vs BBR on "
            "a 100ms-RTT bottleneck link",
            2,
            "simulation reports throughput, utilization, and queueing delay for "
            "both algorithms on 3 bandwidth scenarios",
            "research", "congestion-control", "simulation"),
        challenge(
            "Write a TCP retransmission simulator (RTO estimation per RFC 6298) "
            "and measure spurious retransmits under jitter",
            2,
            "script reports RTO convergence and <2% spurious retransmits on a "
            "jittered 50ms link",
            "code", "tcp", "retransmission"),
        challenge(
            "Implement a NAT traversal demo: UDP hole punching between two "
            "peers via a rendezvous server",
            2,
            "two peers behind simulated NATs establish direct UDP flow; "
            "rendezvous logs the punch sequence",
            "code", "nat", "p2p"),
        challenge(
            "Build a userspace TCP stack fragment: implement the full state "
            "machine with retransmission, SACK, and window scaling over raw loopback",
            3,
            "stack completes a 10MB transfer over loopback with 5% injected "
            "loss, checksum-valid, in-order",
            "build", "tcp", "userspace-stack"),
        challenge(
            "Implement a BGP-style path-vector routing simulator for 20 ASes "
            "and demonstrate loop prevention plus convergence after a link cut",
            3,
            "simulation converges to loop-free paths within 60 update rounds "
            "after cutting 3 links",
            "code", "bgp", "routing"),
        challenge(
            "Build a QUIC 0-RTT / 1-RTT handshake simulator with key derivation "
            "and prove forward secrecy on key rotation",
            3,
            "pytest passes: handshake completes in 1 RTT, rotated keys decrypt "
            "new data, old keys cannot",
            "code", "quic", "cryptography"),
        challenge(
            "Implement a packet-filter VM (bytecode interpreter with verifier, "
            "eBPF-style) and benchmark vs naive filtering",
            3,
            "verifier rejects 5 unsafe programs; VM filters 1M packets ≥10x "
            "faster than a Python loop",
            "code", "ebpf", "packet-filter"),
    ],
    "compilers": [
        challenge(
            "Write a lexer for a tiny expression language (numbers, idents, "
            "+ - * / parens) with exact line/col error reporting",
            1,
            "pytest passes: 50 token streams tokenize correctly; 10 malformed "
            "inputs report exact line:col",
            "code", "lexer", "tokenizing"),
        challenge(
            "Implement a recursive-descent parser producing an AST for the "
            "expression language, with operator precedence",
            1,
            "pytest passes: 40 expressions parse to expected ASTs including "
            "precedence and associativity",
            "code", "parsing", "ast"),
        challenge(
            "Build a pretty-printer that round-trips AST → source → AST for "
            "the expression language",
            1,
            "round-trip passes on 200 generated expressions: ASTs structurally "
            "equal",
            "code", "pretty-printing", "ast"),
        challenge(
            "Implement a Hindley-Milner type inferrer for a lambda calculus "
            "with let-polymorphism",
            2,
            "pytest passes: 30 programs infer principal types; 8 ill-typed "
            "programs rejected with locations",
            "code", "type-inference", "hindley-milner"),
        challenge(
            "Build a bytecode compiler plus stack VM for the expression "
            "language with 12 opcodes",
            2,
            "pytest passes: 100 programs evaluate identically to a tree-walking "
            "interpreter",
            "code", "bytecode", "vm"),
        challenge(
            "Implement constant folding, dead-code elimination, and copy "
            "propagation as AST-to-AST passes",
            2,
            "optimizer reduces instruction count ≥30% on 5 fixtures; semantics "
            "preserved via differential test",
            "code", "optimization", "passes"),
        challenge(
            "Write an SSA construction pass (dominance frontiers, phi "
            "insertion) for a toy CFG",
            2,
            "pytest passes: SSA form validates (single assignment, dominance) "
            "on 6 CFGs incl. loops",
            "code", "ssa", "cfg"),
        challenge(
            "Implement a register allocator with graph coloring (spill when "
            ">4 registers) for straight-line code",
            2,
            "allocator produces valid coloring on 10 fixtures; spills inserted "
            "where needed, liveness verified",
            "code", "register-allocation", "graph-coloring"),
        challenge(
            "Build a compile pipeline for a subset of the language to x86-64 "
            "via a simple codegen and run the binaries",
            3,
            "generated binaries compute fib(20) and 5 other programs correctly "
            "under a 5s timeout",
            "build", "codegen", "x86-64"),
        challenge(
            "Implement a tracing JIT for the stack VM: record hot loops, "
            "compile traces, guard and deopt",
            3,
            "benchmark reports ≥3x speedup on a hot loop; deopt path verified "
            "by a type-changing input",
            "code", "jit", "tracing"),
        challenge(
            "Write an LR(1) parser generator: build canonical item sets and "
            "parse tables from a grammar spec",
            3,
            "generator builds conflict-free tables for the expression grammar; "
            "3 ambiguous grammars report exact conflicts",
            "code", "lr-parsing", "parser-generator"),
        challenge(
            "Implement escape analysis plus inlining for the toy language and "
            "measure allocation reduction",
            3,
            "analysis stack-allocates ≥50% of closures on 4 fixtures; "
            "differential test proves semantics preserved",
            "code", "escape-analysis", "inlining"),
    ],
    "runtimes": [
        challenge(
            "Implement a mark-sweep garbage collector for a toy object graph "
            "and measure pause time vs heap size",
            1,
            "script collects unreachable cycles correctly; pause scales "
            "linearly, reported for 10k/100k objects",
            "code", "gc", "mark-sweep"),
        challenge(
            "Build a single-threaded event loop (timers + I/O callbacks) and "
            "run 1k concurrent fake connections",
            1,
            "loop handles 1k scheduled timers with ≤5ms drift and all callbacks "
            "fire",
            "code", "event-loop", "async"),
        challenge(
            "Implement a bump-allocator arena and benchmark allocation speed "
            "vs malloc",
            1,
            "benchmark reports arena ≥10x faster than per-object malloc for 1M "
            "small allocs",
            "code", "allocator", "arena"),
        challenge(
            "Implement a generational GC (nursery + tenured, minor/major "
            "collections) and tune the nursery size",
            2,
            "script reports promotion rate and pause times; minor GC reclaims "
            "≥90% of short-lived garbage",
            "code", "gc", "generational"),
        challenge(
            "Build stackful coroutines with explicit stack switching plus a "
            "fair scheduler",
            2,
            "pytest passes: 10k coroutines yield/resume correctly; scheduler "
            "fairness within 2x",
            "code", "coroutines", "scheduling"),
        challenge(
            "Implement a JIT basics demo: a template compiler that specializes "
            "an integer adder for observed types",
            2,
            "specialized adder runs ≥5x faster than generic dispatch on 1M "
            "calls; deopt on new type verified",
            "code", "jit", "specialization"),
        challenge(
            "Write an event-loop I/O benchmark: epoll-style reactor vs "
            "thread-per-connection for 5k idle connections",
            2,
            "benchmark reports reactor memory ≤10% of thread-per-connection at "
            "5k connections",
            "research", "event-loop", "scalability"),
        challenge(
            "Implement a copying (Cheney) collector and prove it compacts the "
            "heap with zero fragmentation",
            2,
            "script shows post-GC heap fully compacted: live objects contiguous, "
            "reported occupancy 100%",
            "code", "gc", "copying"),
        challenge(
            "Implement a concurrent tri-color marking collector with SATB "
            "barriers and prove no lost objects under mutation",
            3,
            "pytest passes: 30s mutator stress with barriers — zero lost live "
            "objects, verified by shadow marking",
            "code", "gc", "concurrent"),
        challenge(
            "Build a work-stealing async runtime (executor + reactor + waker) "
            "that passes a subset of a task test-suite",
            3,
            "runtime passes 40 ported task tests (spawn/join/timeout/select) "
            "with no hangs",
            "build", "async-runtime", "work-stealing"),
        challenge(
            "Implement on-stack replacement (OSR): detect a hot loop in the "
            "interpreter and swap to compiled code mid-flight",
            3,
            "demo shows OSR trigger at 10k iterations; compiled continuation "
            "produces identical results",
            "code", "jit", "osr"),
        challenge(
            "Build a precise GC with stack maps: generate stack maps for "
            "compiled frames and collect with exact roots",
            3,
            "collector runs 5k collections with zero false roots retained "
            "(verified against a liveness oracle)",
            "code", "gc", "stack-maps"),
    ],
    "packaging": [
        challenge(
            "Write a PEP 440 version parser plus comparator that orders 30 "
            "tricky versions (rc, dev, post, epochs)",
            1,
            "pytest passes: 30 versions sort in the exact order the `packaging` "
            "library produces",
            "code", "pep440", "versioning"),
        challenge(
            "Build a wheel inspector that lists a .whl's METADATA, WHEEL, "
            "RECORD and verifies RECORD hashes",
            1,
            "inspector validates 3 real wheels; 1 tampered wheel flagged with "
            "the mismatched path",
            "build", "wheel", "integrity"),
        challenge(
            "Implement a lockfile generator for a toy package index that pins "
            "exact versions plus hashes",
            1,
            "generator emits lockfile; verifier confirms all pins resolve to "
            "identical hashes on re-run",
            "code", "lockfile", "reproducibility"),
        challenge(
            "Implement a backtracking dependency resolver (PubGrub-style "
            "version solving) over a synthetic index",
            2,
            "pytest passes: 20 resolution fixtures incl. conflicts produce "
            "correct solutions or precise conflict reports",
            "code", "resolver", "pubgrub"),
        challenge(
            "Build a minimal wheel builder: package a pure-Python project "
            "into an installable .whl",
            2,
            "built wheel installs with pip into a fresh venv and `import` works; "
            "RECORD hashes verify",
            "build", "wheel", "builder"),
        challenge(
            "Implement an SBOM generator (CycloneDX) for a requirements file "
            "with license detection",
            2,
            "generator emits valid CycloneDX JSON (schema-checked) listing all "
            "15 deps with licenses",
            "build", "sbom", "cyclonedx"),
        challenge(
            "Write a reproducible-build verifier: build the same sdist twice "
            "and prove byte-identical output",
            2,
            "two builds produce identical sha256; verifier reports normalized "
            "timestamps and permissions",
            "code", "reproducible-builds", "sdist"),
        challenge(
            "Implement a vulnerability scanner that matches a lockfile against "
            "the OSV database format",
            2,
            "scanner flags all 5 planted CVEs in a fixture lockfile with zero "
            "false positives",
            "code", "vulnerability", "osv"),
        challenge(
            "Implement a full SAT-based resolver with conflict-driven learning "
            "over a 10k-package synthetic index",
            3,
            "resolver solves a 500-dependency fixture in <5s; 10 unsatisfiable "
            "fixtures yield minimal conflict sets",
            "code", "resolver", "sat"),
        challenge(
            "Build a binary wheel audit tool: scan .so files for manylinux "
            "compatibility (glibc symbols, bundled libs)",
            3,
            "audit correctly classifies 5 fixture wheels (manylinux2014 vs "
            "incompatible) with symbol-level reasons",
            "build", "manylinux", "audit"),
        challenge(
            "Implement a supply-chain provenance verifier: verify SLSA-style "
            "attestations and Sigstore bundles for artifacts",
            3,
            "verifier accepts 3 valid bundles and rejects tampered payload, "
            "expired cert, and wrong-subject bundles",
            "code", "slsa", "provenance"),
        challenge(
            "Build a hermetic offline installer: resolve, fetch, and install a "
            "dependency closure with no network at install time",
            3,
            "install of a 20-package closure succeeds with network disabled; "
            "import smoke test passes",
            "build", "offline-install", "hermetic"),
    ],
}

register_topic_pack("chall_systems", _PACK)
