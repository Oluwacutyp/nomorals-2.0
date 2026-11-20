"""Xsens MVN wire: packet encode/decode, receiver/sender/replayer,
pcap reading, and the mocap-tool / CLI / chat wiring.

The wire format is the Xsens MVN Studio real-time network streaming
protocol (MV0305P Rev E): 24-byte big-endian header, pose types 01
(Euler, Y-up) and 02 (quaternion, Z-up), character info 13 (name
table). Fixture frames are synthetic, built in-test; the real
captures in the clone are accepted by prove.py in
~/workspace/xsens-verify/.
"""

import struct
import unittest
from contextlib import suppress
from pathlib import Path

from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin
from nomorals.cmdline.commands.mocap import _cmd_mocap
from nomorals.cmdline.parser import _parser
from nomorals.media.mocap.bvh import (
    Joint,
    Skeleton,
    SkeletonFrame,
    hierarchy_joints,
)
from nomorals.media.mocap.errors import MocapError
from nomorals.media.mocap.mvn import (
    MVN_DEFAULT_PORT,
    MVN_NAMES,
    MVN_SEGMENT_IDS,
    MvnAssembler,
    MvnEulerItem,
    MvnHeader,
    MvnPointItem,
    MvnReceiver,
    MvnReplayer,
    MvnSegmentItem,
    MvnSender,
    decode_datagram,
    decode_header,
    decode_name_table,
    encode_datagram,
    encode_header,
    forward_kinematics,
    frame_to_mvn_items,
    get_sender,
    get_session,
    iter_pcap,
    list_sessions,
    mvn_pose_to_frame,
    mvn_skeleton,
    mvn_to_devon_pos,
    mvn_to_devon_quat,
    name_for_id,
    read_pcap,
    start_receiver,
    start_sender,
    stop_session,
)
from nomorals.media.mocap.state import MocapStateStore
from nomorals.tools.mocap import _mvn_action


def _f32(value: float) -> float:
    """A float through float32 — what the wire stores."""
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _header(msg_type: str = "02", *, sample: int = 0,
            counter: int = 0, last: bool = True, count: int = 0,
            time_ms: int = 0, avatar: int = 0) -> MvnHeader:
    return MvnHeader(
        msg_type=msg_type, sample_counter=sample,
        datagram_counter=counter, last=last, item_count=count,
        time_code_ms=time_ms, avatar_id=avatar)


def _segment(segment_id: int = 1,
             position: tuple[float, float, float] = (0.0, 0.0, 0.0),
             quaternion: tuple[float, float, float, float] = (
                 1.0, 0.0, 0.0, 0.0)) -> MvnSegmentItem:
    return MvnSegmentItem(
        segment_id=segment_id, position=position,
        quaternion=quaternion)


def _table_body(names_offsets) -> bytes:
    """A synthetic ``13`` body: count + (name length, name,
    3 floats) entries."""
    out = bytearray(struct.pack(">I", len(names_offsets)))
    for name, offset in names_offsets:
        raw = name.encode("ascii")
        out += struct.pack(">I", len(raw))
        out += raw
        out += struct.pack(">fff", *offset)
    return bytes(out)


def _ipv4_udp(payload: bytes, *, src_port: int = 9763,
              dst_port: int = 9763) -> bytes:
    """A loopback packet: 4-byte NULL header + IPv4 + UDP."""
    udp_len = 8 + len(payload)
    udp = struct.pack(">HHHH", src_port, dst_port, udp_len, 0)
    total = 20 + udp_len
    ip = struct.pack(
        ">BBHHHBBH4s4s", 0x45, 0, total, 0, 0, 64, 17, 0,
        bytes((127, 0, 0, 1)), bytes((127, 0, 0, 1)))
    return b"\x02\x00\x00\x00" + ip + udp + payload


def _pcap_bytes(payloads, *, network: int = 0) -> bytes:
    """A legacy pcap (microseconds) holding the payloads as
    UDP packets. ``network`` 0 = NULL, 1 = Ethernet."""
    frames = []
    for payload in payloads:
        if network == 1:
            link = (b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"\x08\x00")
            frames.append(link + _ipv4_udp(payload)[4:])
        else:
            frames.append(_ipv4_udp(payload))
    return _pcap_frames(frames, network=network)


def _pcap_frames(frames, *, network: int = 0) -> bytes:
    """A legacy pcap holding pre-built link frames."""
    out = bytearray(struct.pack(
        "<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, network))
    for index, raw in enumerate(frames):
        out += struct.pack("<IIII", 1700000000 + index, 0,
                           len(raw), len(raw))
        out += raw
    return bytes(out)


def _two_joint_skeleton() -> Skeleton:
    """Root + one child, offsets in metres, Y-up."""
    child = Joint(name="Child", offset=(0.0, 0.5, 0.0),
                  channels=("Zrotation", "Xrotation", "Yrotation"))
    root = Joint(name="Root", offset=(0.0, 0.0, 0.0),
                 channels=("Xposition", "Yposition", "Zposition",
                           "Zrotation", "Xrotation", "Yrotation"),
                 children=[child])
    return Skeleton(root=root, frame_time=1.0 / 120.0)


class CodecTest(unittest.TestCase):
    def test_header_round_trip(self):
        header = _header("02", sample=1660, counter=0, last=True,
                         count=63, time_ms=6916, avatar=2)
        decoded = decode_header(encode_header(header, body_len=2016))
        self.assertEqual(decoded, header)

    def test_header_single_datagram_flag(self):
        header = _header("02", counter=0, last=True, count=63)
        raw = encode_header(header)
        self.assertEqual(raw[10], 0x80)
        self.assertEqual(decode_header(raw).datagram_counter, 0)
        self.assertTrue(decode_header(raw).last)

    def test_header_split_datagram_counters(self):
        for counter, last, raw_byte in ((0, False, 0x00),
                                        (1, False, 0x01),
                                        (2, True, 0x82)):
            header = _header("13", counter=counter, last=last)
            self.assertEqual(encode_header(header)[10], raw_byte)

    def test_header_reserved_bytes_carry_body_length(self):
        header = _header("02", count=63)
        raw = encode_header(header, body_len=2016)
        self.assertEqual(raw[17:20], b"\x17\x00\x28")
        self.assertEqual(raw[20:24], struct.pack(">I", 2016))

    def test_02_round_trip(self):
        items = (
            _segment(1, (-0.10, 0.61, 0.82),
                     (0.689, 0.003, -0.037, -0.724)),
            _segment(7, (1.0, 2.0, 3.0),
                     (0.7071, 0.0, 0.7071, 0.0)),
            _segment(63, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        )
        data = encode_datagram(_header("02", count=3), items)
        decoded = decode_datagram(data)
        self.assertEqual(len(decoded.items), 3)
        for got, want in zip(decoded.items, items, strict=True):
            self.assertEqual(got.segment_id, want.segment_id)
            for g, w in zip(got.position, want.position, strict=True):
                self.assertAlmostEqual(g, _f32(w), places=5)
            for g, w in zip(got.quaternion, want.quaternion, strict=True):
                self.assertAlmostEqual(g, _f32(w), places=5)

    def test_01_round_trip(self):
        items = (MvnEulerItem(segment_id=1, position=(1.0, 2.0, 3.0),
                              euler=(10.0, 20.0, 30.0)),)
        data = encode_datagram(_header("01", count=1), items)
        decoded = decode_datagram(data)
        self.assertEqual(decoded.items[0].euler[0], _f32(10.0))

    def test_03_and_04_round_trip(self):
        for msg_type in ("03", "04"):
            items = (MvnPointItem(point_id=113,
                                  position=(1.0, 2.0, 3.0)),)
            data = encode_datagram(_header(msg_type, count=1), items)
            decoded = decode_datagram(data)
            self.assertEqual(decoded.items[0].point_id, 113)
            self.assertAlmostEqual(
                decoded.items[0].position[2], 3.0, places=5)

    def test_10_round_trip(self):
        item = {"segment_id": 1, "offset": (0.1, 0.2, 0.3),
                "matrix": tuple(float(i) for i in range(9))}
        data = encode_datagram(_header("10", count=1), (item,))
        decoded = decode_datagram(data)
        self.assertEqual(decoded.items[0]["segment_id"], 1)
        self.assertAlmostEqual(
            decoded.items[0]["matrix"][8], 8.0, places=5)

    def test_11_round_trip(self):
        items = ({"segment_id": 25, "name": "Ball"},)
        data = encode_datagram(_header("11", count=1), items)
        decoded = decode_datagram(data)
        self.assertEqual(decoded.items[0]["name"], "Ball")

    def test_12_and_13_raw_body_round_trip(self):
        for msg_type, body in (
                ("12", b"name: Dancer\nxmid: 42\n"),
                ("13", _table_body([("Pelvis", (0.0, 0.0, 0.913))]))):
            data = encode_datagram(_header(msg_type), body=body)
            decoded = decode_datagram(data)
            self.assertEqual(decoded.body, body)

    def test_decode_name_table_small(self):
        body = _table_body([
            ("Pelvis", (0.0, 0.0, 0.913)),
            ("L5", (-0.0108, 0.0, 1.0105)),
            ("L3", (-0.0108, 0.0, 1.0800)),
        ])
        table = decode_name_table(body)
        self.assertEqual(table.names,
                         ("Pelvis", "L5", "L3"))
        self.assertEqual(table.consumed, len(body))
        self.assertEqual(table.extra, b"")
        self.assertAlmostEqual(table.offsets_m[0][2], 0.913,
                               places=5)

    def test_decode_name_table_keeps_extra_and_names(self):
        body = (_table_body([("Pelvis", (0.0, 0.0, 0.913))])
                + struct.pack(">I", 10) + b"pHipOrigin"
                + struct.pack(">I", 5) + b"jL5S1"
                + b"\xbc\xaa\xf9\xe1")
        table = decode_name_table(body)
        self.assertEqual(table.names, ("Pelvis",))
        self.assertIn("pHipOrigin", table.extra_names)
        self.assertIn("jL5S1", table.extra_names)
        self.assertGreater(len(table.extra), 0)

    def test_refuse_bad_magic(self):
        data = encode_datagram(_header("02", count=1),
                               (_segment(1),))
        with self.assertRaises(MocapError) as caught:
            decode_datagram(b"XXXX" + data[4:])
        self.assertEqual(caught.exception.kind, "bad_magic")

    def test_refuse_short_datagram(self):
        with self.assertRaises(MocapError) as caught:
            decode_datagram(b"MXTP02" + b"\x00" * 5)
        self.assertEqual(caught.exception.kind, "short_datagram")

    def test_refuse_length_mismatch(self):
        data = encode_datagram(_header("02", count=2),
                               (_segment(1), _segment(2)))
        truncated = data[:-32]
        with self.assertRaises(MocapError) as caught:
            decode_datagram(truncated)
        self.assertEqual(caught.exception.kind, "length_mismatch")

    def test_refuse_non_normalised_quaternion(self):
        item = _segment(1, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0, 0.0))
        with self.assertRaises(MocapError) as caught:
            encode_datagram(_header("02", count=1), (item,))
        self.assertEqual(caught.exception.kind, "bad_quaternion")

    def test_refuse_unknown_type(self):
        header = _header("99", count=0)
        raw = encode_header(header) + b""
        with self.assertRaises(MocapError) as caught:
            decode_datagram(raw)
        self.assertEqual(caught.exception.kind, "unknown_type")

    def test_refuse_type_05(self):
        raw = encode_header(_header("05")) + b"\x00" * 4
        with self.assertRaises(MocapError) as caught:
            decode_datagram(raw)
        self.assertEqual(caught.exception.kind, "unsupported_type")

    def test_encode_refuses_item_count_mismatch(self):
        with self.assertRaises(MocapError) as caught:
            encode_datagram(_header("02", count=2), (_segment(1),))
        self.assertEqual(caught.exception.kind, "length_mismatch")


class NamesTest(unittest.TestCase):
    def test_name_for_id_wire_default(self):
        self.assertEqual(name_for_id(1), "Pelvis")
        self.assertEqual(name_for_id(22), "LeftFoot")
        self.assertEqual(name_for_id(23), "LeftToe")
        self.assertEqual(name_for_id(63), "RightFifthDP")

    def test_name_for_id_legacy_spec_table(self):
        self.assertEqual(name_for_id(23, legacy=True), "LeftFoot")
        self.assertEqual(name_for_id(24, legacy=True), "LeftToe")
        self.assertEqual(name_for_id(25, legacy=True), "Prop1")
        with self.assertRaises(MocapError):
            name_for_id(22, legacy=True)

    def test_name_for_id_live_table_wins(self):
        table = decode_name_table(_table_body([
            ("Custom", (0.0, 0.0, 1.0)),
            ("Other", (0.0, 0.0, 1.1)),
        ]))
        self.assertEqual(name_for_id(1, table=table), "Custom")
        self.assertEqual(name_for_id(2, table=table), "Other")
        with self.assertRaises(MocapError):
            name_for_id(3, table=table)

    def test_name_for_id_unknown_refused(self):
        with self.assertRaises(MocapError) as caught:
            name_for_id(64)
        self.assertEqual(caught.exception.kind, "unknown_segment")
        self.assertEqual(name_for_id(1), MVN_NAMES[0])
        self.assertEqual(len(MVN_SEGMENT_IDS), 27)


class MathTest(unittest.TestCase):
    def test_euler_quat_round_trip(self):
        from nomorals.media.mocap.mvn import (
            euler_zxy_to_quat,
            quat_to_euler_zxy,
        )
        for angles in ((0.0, 0.0, 0.0), (30.0, -45.0, 60.0),
                       (170.0, 10.0, -170.0), (-90.0, 20.0, 90.0)):
            quat = euler_zxy_to_quat(*angles)
            back = quat_to_euler_zxy(quat)
            requat = euler_zxy_to_quat(*back)
            for got, want in zip(requat, quat, strict=True):
                self.assertAlmostEqual(got, want, places=6)

    def test_euler_quat_gimbal(self):
        from nomorals.media.mocap.mvn import (
            euler_zxy_to_quat,
            quat_to_euler_zxy,
        )
        quat = euler_zxy_to_quat(30.0, 90.0, 40.0)
        back = quat_to_euler_zxy(quat)
        requat = euler_zxy_to_quat(*back)
        for got, want in zip(requat, quat, strict=True):
            self.assertAlmostEqual(got, want, places=6)

    def test_axis_conversion_position(self):
        # MVN (x, y, z) cm Z-up -> Devon (x, z, -y) metres Y-up.
        self.assertEqual(mvn_to_devon_pos((0.0, 100.0, 0.0)),
                         (0.0, 0.0, -1.0))
        self.assertEqual(mvn_to_devon_pos((10.0, 20.0, 30.0)),
                         (0.1, 0.3, -0.2))

    def test_axis_conversion_quat_round_trip(self):
        from nomorals.media.mocap.mvn import devon_to_mvn_quat
        quat = (0.689, 0.003, -0.037, -0.724)
        back = devon_to_mvn_quat(mvn_to_devon_quat(quat))
        for got, want in zip(back, quat, strict=True):
            self.assertAlmostEqual(got, want, places=12)

    def test_forward_kinematics_simple(self):
        skeleton = _two_joint_skeleton()
        frame = SkeletonFrame(values={
            "Root": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            "Child": (0.0, 0.0, 0.0)})
        world = forward_kinematics(skeleton, frame)
        self.assertEqual(world["Root"]["position"], (0.0, 1.0, 0.0))
        self.assertEqual(world["Child"]["position"], (0.0, 1.5, 0.0))

    def test_forward_kinematics_rotated_root(self):
        skeleton = _two_joint_skeleton()
        frame = SkeletonFrame(values={
            "Root": (0.0, 1.0, 0.0, 90.0, 0.0, 0.0),
            "Child": (0.0, 0.0, 0.0)})
        world = forward_kinematics(skeleton, frame)
        pos = world["Child"]["position"]
        self.assertAlmostEqual(pos[0], -0.5, places=6)
        self.assertAlmostEqual(pos[1], 1.0, places=6)
        self.assertAlmostEqual(pos[2], 0.0, places=6)

    def test_forward_kinematics_refuses_missing_joint(self):
        skeleton = _two_joint_skeleton()
        frame = SkeletonFrame(values={
            "Root": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)})
        with self.assertRaises(MocapError):
            forward_kinematics(skeleton, frame)


class SkeletonTest(unittest.TestCase):
    def test_mvn_skeleton_shape(self):
        skeleton = mvn_skeleton()
        joints = list(hierarchy_joints(skeleton))
        self.assertEqual(len(joints), 63)
        self.assertEqual(skeleton.root.name, "Pelvis")
        self.assertEqual(skeleton.root.channels,
                         ("Xposition", "Yposition", "Zposition",
                          "Zrotation", "Xrotation", "Yrotation"))
        by_name = {joint.name: joint for joint in joints}
        self.assertEqual(by_name["Head"].channels,
                         ("Zrotation", "Xrotation", "Yrotation"))
        # Head's parent is Neck: its offset is Neck-relative.
        head = by_name["Head"]
        self.assertAlmostEqual(head.offset[1], 0.0916, places=3)

    def test_mvn_skeleton_offsets_parent_relative(self):
        skeleton = mvn_skeleton()
        by_name = {j.name: j for j in hierarchy_joints(skeleton)}
        # L5 sits 0.0975 m above the pelvis in the table.
        self.assertAlmostEqual(by_name["L5"].offset[1], 0.0975,
                               places=4)
        # LeftFoot is the wire order's 22nd entry, not the spec's.
        self.assertAlmostEqual(by_name["LeftFoot"].offset[1], -0.4059,
                               places=4)

    def test_mvn_pose_to_frame(self):
        items = (
            _segment(1, (10.0, 20.0, 30.0),
                     (1.0, 0.0, 0.0, 0.0)),
            _segment(7, (0.0, 0.0, 0.0),
                     (0.7071, 0.0, 0.0, 0.7071)),
        )
        frame = mvn_pose_to_frame(items)
        self.assertEqual(len(frame.values["Pelvis"]), 6)
        self.assertAlmostEqual(frame.values["Pelvis"][0], 0.1,
                               places=5)
        self.assertAlmostEqual(frame.values["Pelvis"][1], 0.3,
                               places=5)
        self.assertEqual(len(frame.values["Head"]), 3)
        self.assertEqual(frame.unmapped, [])

    def test_mvn_pose_to_frame_unmapped_counted(self):
        frame = mvn_pose_to_frame((_segment(64),))
        self.assertEqual(frame.values, {})
        self.assertEqual(frame.unmapped, ["id:64"])

    def test_frame_to_mvn_items_round_trip(self):
        skeleton = mvn_skeleton()
        frame = SkeletonFrame(values={
            joint.name: ((0.0, 0.95, 0.0, 0.0, 0.0, 0.0)
                         if joint is skeleton.root
                         else (0.0, 0.0, 0.0))
            for joint in hierarchy_joints(skeleton)})
        items, unmapped = frame_to_mvn_items(skeleton, frame)
        self.assertEqual(unmapped, [])
        self.assertEqual(len(items), 63)
        pelvis = next(i for i in items if i.segment_id == 1)
        self.assertAlmostEqual(pelvis.position[0], 0.0, places=5)
        self.assertAlmostEqual(pelvis.position[2], 95.0, places=5)


if __name__ == "__main__":
    unittest.main()


class PcapTest(unittest.TestCase):
    def test_iter_pcap_null_linktype(self):
        payloads = [
            encode_datagram(_header("02", count=1), (_segment(1),)),
            encode_datagram(_header("02", count=1), (_segment(2),)),
        ]
        packets, counts = iter_pcap(_pcap_bytes(payloads))
        self.assertEqual(counts["packets"], 2)
        self.assertEqual(counts["non_ip"], 0)
        self.assertEqual(counts["non_udp"], 0)
        self.assertEqual(len(packets), 2)
        self.assertEqual(packets[0].payload, payloads[0])
        self.assertEqual(packets[0].src, "127.0.0.1")
        self.assertEqual(packets[0].dst_port, 9763)

    def test_iter_pcap_ethernet_linktype(self):
        payload = encode_datagram(_header("02", count=1),
                                  (_segment(1),))
        packets, counts = iter_pcap(
            _pcap_bytes([payload], network=1))
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0].payload, payload)

    def test_iter_pcap_skips_non_udp(self):
        udp = _ipv4_udp(b"hello")
        tcp = bytearray(_ipv4_udp(b"hello"))
        tcp[13] = 6  # IP protocol byte: TCP, not UDP
        packets, counts = iter_pcap(
            _pcap_frames([udp, bytes(tcp)]))
        self.assertEqual(len(packets), 1)
        self.assertEqual(counts["non_udp"], 1)

    def test_iter_pcap_refuses_bad_magic(self):
        with self.assertRaises(MocapError) as caught:
            iter_pcap(b"NOPE" + b"\x00" * 40)
        self.assertEqual(caught.exception.kind, "bad_magic")

    def test_iter_pcap_refuses_unsupported_linktype(self):
        blob = bytearray(_pcap_bytes([]))
        blob[20:24] = struct.pack("<I", 999)
        with self.assertRaises(MocapError) as caught:
            iter_pcap(bytes(blob))
        self.assertEqual(caught.exception.kind,
                         "unsupported_linktype")

    def test_read_pcap_missing_file(self):
        with self.assertRaises(MocapError) as caught:
            read_pcap("no-such-dir/mvn.pcap")
        self.assertEqual(caught.exception.kind, "missing_pcap")


class AssemblerTest(unittest.TestCase):
    def _split_13(self):
        """A 3-part ``13`` message: two plain parts + the tail."""
        body = _table_body([("Pelvis", (0.0, 0.0, 0.913)),
                            ("L5", (0.0, 0.0, 1.0))])
        part0 = body[:20]
        part1 = body[20:40]
        part2 = body[40:]
        return part0, part1, part2

    def test_merges_split_13_with_0x80_tail(self):
        part0, part1, part2 = self._split_13()
        assembler = MvnAssembler()
        self.assertEqual(assembler.feed(encode_datagram(
            _header("13", counter=0, last=False), body=part0)), [])
        self.assertEqual(assembler.feed(encode_datagram(
            _header("13", counter=1, last=False), body=part1)), [])
        merged = assembler.feed(encode_datagram(
            _header("13", counter=0, last=True), body=part2))
        self.assertEqual(len(merged), 1)
        table = decode_name_table(merged[0].body)
        self.assertEqual(table.names, ("Pelvis", "L5"))

    def test_merges_spec_conformant_tail(self):
        part0, part1, part2 = self._split_13()
        assembler = MvnAssembler()
        assembler.feed(encode_datagram(
            _header("13", counter=0, last=False), body=part0))
        assembler.feed(encode_datagram(
            _header("13", counter=1, last=False), body=part1))
        merged = assembler.feed(encode_datagram(
            _header("13", counter=2, last=True), body=part2))
        table = decode_name_table(merged[0].body)
        self.assertEqual(table.names, ("Pelvis", "L5"))

    def test_refuses_incomplete_sample(self):
        part0, _part1, part2 = self._split_13()
        assembler = MvnAssembler()
        assembler.feed(encode_datagram(
            _header("13", counter=0, last=False), body=part0))
        with self.assertRaises(MocapError) as caught:
            assembler.feed(encode_datagram(
                _header("13", counter=2, last=True), body=part2))
        self.assertEqual(caught.exception.kind, "incomplete_sample")


class _FakeSocket:
    """A UDP socket stand-in: canned inbound datagrams, a send
    log, and a close flag."""

    def __init__(self, inbound=()):
        self._inbound = list(inbound)
        self.sent = []
        self.closed = False
        self.timeout = None

    def settimeout(self, value):
        self.timeout = value

    def recvfrom(self, _size):
        if self._inbound:
            return self._inbound.pop(0), ("127.0.0.1", 9763)
        raise TimeoutError("timed out")

    def send(self, data):
        self.sent.append(data)
        return len(data)

    def connect(self, _addr):
        return None

    def bind(self, _addr):
        return None

    def close(self):
        self.closed = True


class ReceiverSenderTest(unittest.TestCase):
    def _feed_13_set(self, receiver):
        part0 = _table_body([("Pelvis", (0.0, 0.0, 0.913)),
                             ("L5", (0.0, 0.0, 1.0))])
        receiver.feed(encode_datagram(
            _header("13", counter=0, last=False), body=part0[:20]))
        receiver.feed(encode_datagram(
            _header("13", counter=1, last=False),
            body=part0[20:40]))
        receiver.feed(encode_datagram(
            _header("13", counter=0, last=True),
            body=part0[40:]))

    def test_receiver_02_writes_take(self):
        store = MocapStateStore()
        receiver = MvnReceiver(store)
        receipt = receiver.feed(encode_datagram(
            _header("02", count=2),
            (_segment(1, (10.0, 20.0, 30.0)),
             _segment(7, (0.0, 0.0, 0.0)))))
        self.assertEqual(receiver.counts["frames"], 1)
        self.assertEqual(receiver.counts["dropped"], 0)
        view = store.read("skeleton")
        frame = view["payload"]["frames"][0]
        self.assertEqual(len(frame["Pelvis"]), 6)
        self.assertEqual(len(frame["Head"]), 3)
        self.assertEqual(receipt["messages"][0]["segments"], 2)

    def test_receiver_13_refreshes_table(self):
        store = MocapStateStore()
        receiver = MvnReceiver(store)
        self._feed_13_set(receiver)
        self.assertEqual(receiver.counts["tables"], 1)
        table = receiver.table_for(0)
        self.assertIsNotNone(table)
        self.assertEqual(table.names, ("Pelvis", "L5"))

    def test_receiver_unknown_type_counted_not_fatal(self):
        store = MocapStateStore()
        receiver = MvnReceiver(store)
        receipt = receiver.feed(
            encode_datagram(_header("12"), body=b"name: X\n"))
        self.assertEqual(receiver.counts["dropped"], 1)
        self.assertEqual(
            receiver.dropped_by_reason.get("unhandled_type_12"), 1)
        self.assertIn("dropped", receipt["messages"][0])

    def test_receiver_bad_datagram_counted(self):
        store = MocapStateStore()
        receiver = MvnReceiver(store)
        receipt = receiver.feed(b"XXXX" + b"\x00" * 20)
        self.assertEqual(receiver.counts["dropped"], 1)
        self.assertEqual(receipt["dropped"], "bad_magic")
        self.assertEqual(receiver.counts["datagrams"], 1)

    def test_sender_round_trip_via_fake_socket(self):
        store = MocapStateStore()
        receiver = MvnReceiver(store)
        items = [_segment(1, (10.0, 20.0, 30.0),
                          (0.689, 0.003, -0.037, -0.724))]
        items += [_segment(i) for i in range(2, 64)]
        receiver.feed(encode_datagram(
            _header("02", count=63, sample=7, time_ms=42),
            tuple(items)))
        sent = []
        sender = MvnSender(store, send=lambda data, addr: sent.append(
            (data, addr)))
        receipt = sender.send_frame("127.0.0.1")
        self.assertEqual(receipt["segments"], 63)
        self.assertEqual(receipt["unmapped"], [])
        data, addr = sent[0]
        self.assertEqual(addr, ("127.0.0.1", MVN_DEFAULT_PORT))
        decoded = decode_datagram(data)
        self.assertEqual(decoded.header.msg_type, "02")
        self.assertEqual(len(decoded.items), 63)
        pelvis = next(i for i in decoded.items
                      if i.segment_id == 1)
        # FK over the take reproduces the pelvis position.
        self.assertAlmostEqual(pelvis.position[0], 10.0, places=4)
        self.assertAlmostEqual(pelvis.position[2], 30.0, places=4)

    def test_sender_with_name_map(self):
        store = MocapStateStore()
        store.register_engine("test", produces=("skeleton",))
        skeleton = Skeleton(
            root=Joint(name="Hips", offset=(0.0, 0.0, 0.0),
                       channels=("Xposition", "Yposition", "Zposition",
                                 "Zrotation", "Xrotation", "Yrotation")))
        from nomorals.media.mocap.bvh import skeleton_to_dict
        store.write("skeleton", "test", {
            "skeleton": skeleton_to_dict(skeleton),
            "frames": [{"Hips": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)}]},
            kind="snapshot")
        sent = []
        sender = MvnSender(
            store, send=lambda data, addr: sent.append(data),
            name_map={"Hips": "Pelvis"})
        receipt = sender.send_frame("127.0.0.1")
        self.assertEqual(receipt["segments"], 1)
        decoded = decode_datagram(sent[0])
        self.assertEqual(decoded.items[0].segment_id, 1)

    def test_sender_refuses_no_take(self):
        store = MocapStateStore()
        sender = MvnSender(store, send=lambda data, addr: None)
        with self.assertRaises(MocapError) as caught:
            sender.send_frame("127.0.0.1")
        self.assertEqual(caught.exception.kind, "no_take")

    def test_sender_refuses_no_mappable_joints(self):
        store = MocapStateStore()
        store.register_engine("test", produces=("skeleton",))
        skeleton = Skeleton(
            root=Joint(name="Nope", offset=(0.0, 0.0, 0.0),
                       channels=("Xposition", "Yposition", "Zposition",
                                 "Zrotation", "Xrotation", "Yrotation")))
        from nomorals.media.mocap.bvh import skeleton_to_dict
        store.write("skeleton", "test", {
            "skeleton": skeleton_to_dict(skeleton),
            "frames": [{"Nope": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)}]},
            kind="snapshot")
        sender = MvnSender(store, send=lambda data, addr: None)
        with self.assertRaises(MocapError) as caught:
            sender.send_frame("127.0.0.1")
        self.assertEqual(caught.exception.kind, "no_mappable_joints")

    def test_replayer_sends_all_payloads(self):
        payloads = [
            encode_datagram(_header("02", count=1), (_segment(1),)),
            encode_datagram(_header("02", count=1), (_segment(2),)),
        ]
        fake = _FakeSocket()
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pcap",
                                        delete=False) as handle:
            handle.write(_pcap_bytes(payloads))
            path = handle.name
        try:
            replayer = MvnReplayer(
                path, send=lambda data, addr: fake.send(data),
                pace_ms=0.0, loop_gap_ms=0.0)
            counts = replayer.play_once()
            self.assertEqual(counts["sent"], 2)
            self.assertEqual(fake.sent, payloads)
            self.assertEqual(replayer.pcap_counts["packets"], 2)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_replayer_refuses_empty_pcap(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pcap",
                                        delete=False) as handle:
            handle.write(_pcap_bytes([]))
            path = handle.name
        try:
            with self.assertRaises(MocapError) as caught:
                MvnReplayer(path, send=lambda d, a: None)
            self.assertEqual(caught.exception.kind, "empty_pcap")
        finally:
            Path(path).unlink(missing_ok=True)


class SessionTest(unittest.TestCase):
    def tearDown(self):
        for entry in list_sessions():
            with suppress(MocapError):
                stop_session(entry["name"])

    def test_receiver_session_lifecycle_fake_socket(self):
        fake = _FakeSocket()
        info = start_receiver(
            "lane-test", MocapStateStore(),
            socket_factory=lambda host, port: fake)
        self.assertEqual(info["kind"], "receiver")
        self.assertEqual(info["target"], "udp://0.0.0.0:9763")
        names = [entry["name"] for entry in list_sessions()]
        self.assertIn("lane-test", names)
        self.assertEqual(get_session("lane-test").name, "lane-test")
        stopped = stop_session("lane-test")
        self.assertEqual(stopped["name"], "lane-test")
        self.assertTrue(fake.closed)
        self.assertEqual(list_sessions(), [])

    def test_duplicate_session_refused(self):
        start_receiver("dup", MocapStateStore(),
                       socket_factory=lambda h, p: _FakeSocket())
        with self.assertRaises(MocapError) as caught:
            start_receiver("dup", MocapStateStore(),
                           socket_factory=lambda h, p: _FakeSocket())
        self.assertEqual(caught.exception.kind, "duplicate_session")

    def test_unknown_session_refused(self):
        with self.assertRaises(MocapError) as caught:
            stop_session("no-such-session")
        self.assertEqual(caught.exception.kind, "unknown_session")
        with self.assertRaises(MocapError):
            get_sender("no-such-session")

    def test_start_receiver_refuses_unbindable(self):
        def _bad_factory(host, port):
            raise OSError("sandbox forbids UDP")

        with self.assertRaises(MocapError) as caught:
            start_receiver("bad", MocapStateStore(),
                           socket_factory=_bad_factory)
        self.assertEqual(caught.exception.kind, "udp_unavailable")
        self.assertEqual(list_sessions(), [])

    def test_sender_session_holds_sender(self):
        fake = _FakeSocket()
        start_sender("snd", MocapStateStore(),
                     socket_factory=lambda: fake)
        sender = get_sender("snd")
        self.assertIsInstance(sender, MvnSender)
        stop_session("snd")
        self.assertTrue(fake.closed)


class ToolTest(unittest.TestCase):
    def tearDown(self):
        for entry in list_sessions():
            with suppress(MocapError):
                stop_session(entry["name"])

    def test_tool_mvn_names(self):
        info = _mvn_action("mvn_names", "", {}, [])
        self.assertEqual(len(info["wire_2022"]), 63)
        self.assertEqual(info["wire_2022"][0], {"id": 1,
                                                "name": "Pelvis"})
        self.assertEqual(info["wire_2022"][22],
                         {"id": 23, "name": "LeftToe"})
        self.assertEqual(len(info["spec_23_segment"]), 27)
        self.assertIn({"id": 23, "name": "LeftFoot"},
                      info["spec_23_segment"])

    def test_tool_mvn_decode_synthetic_pcap(self):
        import tempfile
        payloads = [
            encode_datagram(_header("02", count=1), (_segment(1),)),
        ]
        with tempfile.NamedTemporaryFile(suffix=".pcap",
                                        delete=False) as handle:
            handle.write(_pcap_bytes(payloads))
            path = handle.name
        try:
            info = _mvn_action("mvn_decode", "", {}, [path])
            self.assertEqual(info["pcap_counts"]["packets"], 1)
            self.assertEqual(info["frames"], 1)
            self.assertEqual(info["dropped"], 0)
            self.assertEqual(list(info["last_pelvis"][:3]),
                             [0.0, 0.0, 0.0])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_tool_mvn_listen_send_stop_fake_sockets(self):
        import nomorals.media.mocap.mvn as mvn
        import nomorals.tools.mocap as tool_mocap

        fake_recv = _FakeSocket()
        fake_send = _FakeSocket()
        real_factory = mvn._make_udp_socket
        real_socket = mvn.socket.socket
        old_store = tool_mocap._STORE
        mvn._make_udp_socket = lambda host, port: fake_recv
        mvn.socket.socket = lambda *a, **k: fake_send
        try:
            # A full frame must exist before send.
            store = MocapStateStore()
            receiver = MvnReceiver(store)
            items = [_segment(i) for i in range(1, 64)]
            receiver.feed(encode_datagram(
                _header("02", count=63), tuple(items)))
            tool_mocap._STORE = store
            info = _mvn_action("mvn_listen", "tool-test",
                               {"host": "0.0.0.0", "port": 9763}, [])
            self.assertEqual(info["kind"], "receiver")
            receipt = _mvn_action("mvn_send", "tool-send",
                                  {"host": "127.0.0.1",
                                   "port": 9763}, [])
            self.assertEqual(receipt["segments"], 63)
            self.assertEqual(len(fake_send.sent), 1)
            _mvn_action("mvn_stop", "tool-test", {}, [])
            _mvn_action("mvn_stop", "tool-send", {}, [])
            self.assertEqual(list_sessions(), [])
        finally:
            mvn._make_udp_socket = real_factory
            mvn.socket.socket = real_socket
            tool_mocap._STORE = old_store

    def test_tool_unknown_mvn_action(self):
        with self.assertRaises(ValueError):
            _mvn_action("mvn_nope", "", {}, [])


class CliTest(unittest.TestCase):
    def tearDown(self):
        for entry in list_sessions():
            with suppress(MocapError):
                stop_session(entry["name"])

    def _run(self, argv):
        args = _parser().parse_args(argv)
        return _cmd_mocap(args, None)

    def test_cli_mvn_names(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self._run(["mocap", "mvn", "names"])
        self.assertEqual(rc, 0)
        self.assertIn("LeftToe", buf.getvalue())

    def test_cli_mvn_decode(self):
        import io
        import tempfile
        from contextlib import redirect_stdout
        payloads = [encode_datagram(_header("02", count=1),
                                    (_segment(1),))]
        with tempfile.NamedTemporaryFile(suffix=".pcap",
                                        delete=False) as handle:
            handle.write(_pcap_bytes(payloads))
            path = handle.name
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = self._run(["mocap", "mvn", "decode",
                                "--pcap", path])
            self.assertEqual(rc, 0)
            self.assertIn("frames: 1", buf.getvalue())
        finally:
            Path(path).unlink(missing_ok=True)

    def test_cli_mvn_decode_needs_pcap(self):
        import io
        from contextlib import redirect_stderr
        buf = io.StringIO()
        with redirect_stderr(buf):
            rc = self._run(["mocap", "mvn", "decode"])
        self.assertEqual(rc, 2)


class ChatTest(unittest.TestCase):
    def tearDown(self):
        for entry in list_sessions():
            with suppress(MocapError):
                stop_session(entry["name"])

    def test_chat_mvn_names(self):
        mixin = RuntimeMocapMixin()
        reply = mixin._control_mocap("mvn names", "chat-key")
        self.assertIn("63 segments", reply)
        self.assertIn("LeftFoot 23", reply)

    def test_chat_mvn_help_lists_mvn(self):
        mixin = RuntimeMocapMixin()
        self.assertIn("mvn decode",
                      mixin._control_mocap("", "chat-key"))
        self.assertIn("mvn listen",
                      mixin._control_mocap("bogus", "chat-key"))

    def test_chat_mvn_decode(self):
        import tempfile
        payloads = [encode_datagram(_header("02", count=1),
                                    (_segment(1),))]
        with tempfile.NamedTemporaryFile(suffix=".pcap",
                                        delete=False) as handle:
            handle.write(_pcap_bytes(payloads))
            path = handle.name
        try:
            mixin = RuntimeMocapMixin()
            reply = mixin._control_mocap(
                f"mvn decode {path}", "chat-key")
            self.assertIn("1 packets", reply)
            self.assertIn("1 frames", reply)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_chat_mvn_status_empty(self):
        mixin = RuntimeMocapMixin()
        reply = mixin._control_mocap("mvn status", "chat-key")
        self.assertIn("none", reply)


if __name__ == "__main__":
    unittest.main()
