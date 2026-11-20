"""Kinect record/playback — the k4arecord half of the echelon/k4a-sys port (lane C.3, census 3C.3).

The upstream crate only binds k4a.h; the vendored k4arecord headers
(record.h / playback.h / types.h) were never bound by it or by the
temp port (lane 122). This file proves the libk4arecord binding:
structs against gcc's reading of the repo-vendored headers, the
record→playback round trip against a fake libk4a+libk4arecord, and
the mocap-tool / CLI-shaped actions on top.

The fake library is ONE .so compiled by the test (gcc, into
tmp_path) exporting all 65 k4a_* symbols and all 40 k4arecord
symbols, with an in-memory recording registry keyed by path, so a
recording written by the fake can be played back by the fake in the
same process. It proves the Python binding marshals correctly and
the session logic works; it does NOT prove the real SDK's .mkv
format — that needs real libk4arecord, which is not on this box.
The fake returns canonical frames for any capture handle (it does
not store image bytes), so round-trip tests assert presence,
geometry and timestamps, never recorded bytes.
"""

import ctypes
import json
import subprocess
from pathlib import Path

import pytest

from nomorals.hardware.kinect import record as krec
from nomorals.hardware.kinect.device import DeviceConfig, KinectDevice
from nomorals.hardware.kinect.enums import (
    ColorResolution,
    DepthMode,
    ImageFormat,
)
from nomorals.hardware.kinect.errors import NoSdkError
from nomorals.hardware.kinect.frames import Capture, Frame

REPO = Path(__file__).resolve().parents[1]
VENDOR_INCLUDE = (REPO / "nomorals" / "hardware" / "kinect" / "rs"
                  / "k4a-sys" / "vendor" / "include" / "linux")
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "kinect"

FAKE_C = r"""
/* Fake libk4a + libk4arecord, for tests only. One .so exporting both
   symbol sets; the record side keeps an in-memory registry keyed by
   path, so record -> playback works in-process. Canonical frames:
   every capture handle decodes to the same static images. */
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- minimal struct mirrors (layout = the vendored headers) ---- */
typedef struct { int color_format, color_resolution, depth_mode, camera_fps;
    bool synchronized_images_only; int32_t depth_delay_off_color_usec;
    int wired_sync_mode; uint32_t subordinate_delay_off_master_usec;
    bool disable_streaming_indicator; } dev_cfg_t;
typedef struct { float x, y; } f2_t;
typedef struct { float x, y, z; } f3_t;
typedef struct { float temperature; f3_t acc_sample; uint64_t acc_timestamp_usec;
    f3_t gyro_sample; uint64_t gyro_timestamp_usec; } imu_t;
typedef struct { float cx, cy, fx, fy, k1, k2, k3, k4, k5, k6, codx, cody, p2, p1, metric_radius; } params_t;
typedef struct { int type; uint32_t parameter_count; params_t param; } intrinsics_t;
typedef struct { float rotation[9]; float translation[3]; } extr_t;
typedef struct { extr_t extrinsics; intrinsics_t intrinsics; int resolution_width, resolution_height; float metric_radius; } camcal_t;
typedef struct { camcal_t depth_camera_calibration, color_camera_calibration;
    extr_t extrinsics[4][4]; int depth_mode, color_resolution; } cal_t;
typedef struct { int color_format, color_resolution, depth_mode, camera_fps;
    bool color_track_enabled, depth_track_enabled, ir_track_enabled, imu_track_enabled;
    int32_t depth_delay_off_color_usec; int wired_sync_mode;
    uint32_t subordinate_delay_off_master_usec, start_timestamp_offset_usec; } rec_cfg_t;
typedef struct { uint64_t width, height, frame_rate; } vid_settings_t;
typedef struct { bool high_freq_data; } sub_settings_t;

/* ---- fake device state ---- */
static int g_cap_counter = 0;
static int g_imu_counter = 0;
static uint64_t g_cur_ts = 0;
static uint8_t g_color_buf[1024];
static uint16_t g_depth_buf[640 * 576];
static uint16_t g_ir_buf[640 * 576];
static int g_bufs_init = 0;
static void init_bufs(void) {
    if (g_bufs_init) return;
    g_bufs_init = 1;
    g_color_buf[0] = 0xFF; g_color_buf[1] = 0xD8; g_color_buf[2] = 0xFF;
    for (int y = 0; y < 576; y++) for (int x = 0; x < 640; x++) {
        g_depth_buf[y * 640 + x] = (uint16_t)(1000 + (x + y) % 500);
        g_ir_buf[y * 640 + x] = (uint16_t)(2000 + (x * 3 + y) % 300);
    }
}
#define IMG_COLOR ((void*)0x3001)
#define IMG_DEPTH ((void*)0x3002)
#define IMG_IR ((void*)0x3003)

/* ---- k4a core ---- */
uint32_t k4a_device_get_installed_count(void) { return 1; }
int k4a_set_debug_message_handler(void *cb, int lvl, void *ctx) { (void)cb; (void)lvl; (void)ctx; return 0; }
int k4a_set_allocator(void *a, void *f) { (void)a; (void)f; return 0; }
int k4a_device_open(uint32_t idx, void **dev) {
    if (idx != 0) return 1;
    *dev = (void*)0x1001; return 0;
}
void k4a_device_close(void *dev) { (void)dev; }
int k4a_device_get_capture(void *dev, void **cap, int32_t timeout_ms) {
    (void)dev; (void)timeout_ms;
    int n = g_cap_counter++;
    g_cur_ts = (uint64_t)n * 33333;
    *cap = (void*)(uintptr_t)(0x2000 + n);
    return 0;
}
int k4a_device_get_imu_sample(void *dev, imu_t *s, int32_t timeout_ms) {
    (void)dev; (void)timeout_ms;
    int n = g_imu_counter++;
    memset(s, 0, sizeof(*s));
    s->temperature = 36.5f;
    s->acc_sample.x = 0.0f; s->acc_sample.y = 0.0f; s->acc_sample.z = 9.81f;
    s->acc_timestamp_usec = (uint64_t)n * 10000;
    s->gyro_sample.x = 0.1f; s->gyro_sample.y = 0.2f; s->gyro_sample.z = 0.3f;
    s->gyro_timestamp_usec = (uint64_t)n * 10000;
    return 0;
}
int k4a_capture_create(void **cap) { *cap = (void*)(uintptr_t)(0x2000 + (g_cap_counter++)); return 0; }
void k4a_capture_release(void *cap) { (void)cap; }
void k4a_capture_reference(void *cap) { (void)cap; }
void *k4a_capture_get_color_image(void *cap) { (void)cap; init_bufs(); return IMG_COLOR; }
void *k4a_capture_get_depth_image(void *cap) { (void)cap; init_bufs(); return IMG_DEPTH; }
void *k4a_capture_get_ir_image(void *cap) { (void)cap; init_bufs(); return IMG_IR; }
void k4a_capture_set_color_image(void *c, void *i) { (void)c; (void)i; }
void k4a_capture_set_depth_image(void *c, void *i) { (void)c; (void)i; }
void k4a_capture_set_ir_image(void *c, void *i) { (void)c; (void)i; }
void k4a_capture_set_temperature_c(void *c, float t) { (void)c; (void)t; }
float k4a_capture_get_temperature_c(void *c) { (void)c; return 25.0f; }
int k4a_image_create(int fmt, int w, int h, int stride, void **img) {
    (void)fmt; (void)w; (void)h; (void)stride; *img = (void*)(uintptr_t)(0x4000 + (g_cap_counter++)); return 0;
}
int k4a_image_create_from_buffer(int fmt, int w, int h, int stride, void *buf, size_t sz, void *cb, void *ctx, void **img) {
    (void)fmt; (void)w; (void)h; (void)stride; (void)buf; (void)sz; (void)cb; (void)ctx;
    *img = (void*)(uintptr_t)(0x4000 + (g_cap_counter++)); return 0;
}
uint8_t *k4a_image_get_buffer(void *img) {
    init_bufs();
    if (img == IMG_COLOR) return g_color_buf;
    if (img == IMG_DEPTH) return (uint8_t*)g_depth_buf;
    if (img == IMG_IR) return (uint8_t*)g_ir_buf;
    return NULL;
}
size_t k4a_image_get_size(void *img) {
    if (img == IMG_COLOR) return 1024;
    if (img == IMG_DEPTH || img == IMG_IR) return (size_t)640 * 576 * 2;
    return 0;
}
int k4a_image_get_format(void *img) {
    if (img == IMG_COLOR) return 0;
    if (img == IMG_DEPTH) return 4;
    if (img == IMG_IR) return 5;
    return 8;
}
int k4a_image_get_width_pixels(void *img) {
    if (img == IMG_COLOR) return 1280;
    if (img == IMG_DEPTH || img == IMG_IR) return 640;
    return 0;
}
int k4a_image_get_height_pixels(void *img) {
    if (img == IMG_COLOR) return 720;
    if (img == IMG_DEPTH || img == IMG_IR) return 576;
    return 0;
}
int k4a_image_get_stride_bytes(void *img) {
    if (img == IMG_COLOR) return 0;
    if (img == IMG_DEPTH || img == IMG_IR) return 1280;
    return 0;
}
uint64_t k4a_image_get_timestamp_usec(void *img) { (void)img; return g_cur_ts; }
uint64_t k4a_image_get_device_timestamp_usec(void *img) { (void)img; return g_cur_ts; }
uint64_t k4a_image_get_system_timestamp_nsec(void *img) { (void)img; return g_cur_ts * 1000; }
uint64_t k4a_image_get_exposure_usec(void *img) { (void)img; return 10000; }
uint32_t k4a_image_get_white_balance(void *img) { (void)img; return 4500; }
uint32_t k4a_image_get_iso_speed(void *img) { (void)img; return 100; }
void k4a_image_set_device_timestamp_usec(void *i, uint64_t v) { (void)i; (void)v; }
void k4a_image_set_timestamp_usec(void *i, uint64_t v) { (void)i; (void)v; }
void k4a_image_set_system_timestamp_nsec(void *i, uint64_t v) { (void)i; (void)v; }
void k4a_image_set_exposure_usec(void *i, uint64_t v) { (void)i; (void)v; }
void k4a_image_set_exposure_time_usec(void *i, uint64_t v) { (void)i; (void)v; }
void k4a_image_set_white_balance(void *i, uint32_t v) { (void)i; (void)v; }
void k4a_image_set_iso_speed(void *i, uint32_t v) { (void)i; (void)v; }
void k4a_image_reference(void *i) { (void)i; }
void k4a_image_release(void *i) { (void)i; }
int k4a_device_start_cameras(void *dev, const dev_cfg_t *cfg) { (void)dev; (void)cfg; return 0; }
void k4a_device_stop_cameras(void *dev) { (void)dev; }
int k4a_device_start_imu(void *dev) { (void)dev; return 0; }
void k4a_device_stop_imu(void *dev) { (void)dev; }
int k4a_device_get_serialnum(void *dev, char *buf, size_t *size) {
    (void)dev;
    const char *s = "FAKE123456";
    size_t need = strlen(s) + 1;
    if (!buf || *size < need) { *size = need; return 2; }
    memcpy(buf, s, need); *size = need; return 0;
}
typedef struct { uint32_t major, minor, iteration; } ver_t;
typedef struct { ver_t rgb, depth, audio, depth_sensor; int firmware_build, firmware_signature; } hwver_t;
int k4a_device_get_version(void *dev, hwver_t *v) {
    (void)dev;
    v->rgb.major = 1; v->rgb.minor = 6; v->rgb.iteration = 5;
    v->depth = v->rgb; v->audio = v->rgb; v->depth_sensor = v->rgb;
    v->firmware_build = 0; v->firmware_signature = 0; return 0;
}
int k4a_device_get_color_control_capabilities(void *dev, int cmd, bool *auto_, int32_t *mn, int32_t *mx, int32_t *step, int32_t *def, int *mode) {
    (void)dev; (void)cmd; *auto_ = true; *mn = 0; *mx = 100; *step = 1; *def = 50; *mode = 0; return 0;
}
int k4a_device_get_color_control(void *dev, int cmd, int *mode, int32_t *val) { (void)dev; (void)cmd; *mode = 1; *val = 50; return 0; }
int k4a_device_set_color_control(void *dev, int cmd, int mode, int32_t val) { (void)dev; (void)cmd; (void)mode; (void)val; return 0; }
int k4a_device_get_raw_calibration(void *dev, uint8_t *data, size_t *size) { (void)dev; (void)data; (void)size; return 1; }
int k4a_device_get_calibration(void *dev, int dm, int cr, cal_t *cal) {
    (void)dev; memset(cal, 0, sizeof(*cal)); cal->depth_mode = dm; cal->color_resolution = cr; return 0;
}
int k4a_device_get_sync_jack(void *dev, bool *in, bool *out) { (void)dev; *in = false; *out = false; return 0; }
int k4a_calibration_get_from_raw(const char *raw, size_t sz, int dm, int cr, cal_t *cal) {
    (void)raw; (void)sz; memset(cal, 0, sizeof(*cal)); cal->depth_mode = dm; cal->color_resolution = cr; return 0;
}
int k4a_calibration_3d_to_3d(const cal_t *c, const f3_t *src, int st, int tt, f3_t *dst) { (void)c; (void)src; (void)st; (void)tt; (void)dst; return 0; }
int k4a_calibration_2d_to_3d(const cal_t *c, const f2_t *src, float d, int st, int tt, f3_t *dst, int *valid) { (void)c; (void)src; (void)d; (void)st; (void)tt; (void)dst; (void)valid; return 0; }
int k4a_calibration_3d_to_2d(const cal_t *c, const f3_t *src, int st, int tt, f2_t *dst, int *valid) { (void)c; (void)src; (void)st; (void)tt; (void)dst; (void)valid; return 0; }
int k4a_calibration_2d_to_2d(const cal_t *c, const f2_t *src, float d, int st, int tt, f2_t *dst, int *valid) { (void)c; (void)src; (void)d; (void)st; (void)tt; (void)dst; (void)valid; return 0; }
int k4a_calibration_color_2d_to_depth_2d(const cal_t *c, const f2_t *src, void *dep, f2_t *dst, int *valid) { (void)c; (void)src; (void)dep; (void)dst; (void)valid; return 0; }
void *k4a_transformation_create(const cal_t *c) { (void)c; return (void*)0x6001; }
void k4a_transformation_destroy(void *t) { (void)t; }
int k4a_transformation_depth_image_to_color_camera(void *t, void *d, void *c) { (void)t; (void)d; (void)c; return 0; }
int k4a_transformation_depth_image_to_color_camera_custom(void *t, void *d, void *c, void *cu, void *cv, int it, uint32_t iv) { (void)t; (void)d; (void)c; (void)cu; (void)cv; (void)it; (void)iv; return 0; }
int k4a_transformation_color_image_to_depth_camera(void *t, void *d, void *c, void *o) { (void)t; (void)d; (void)c; (void)o; return 0; }
int k4a_transformation_depth_image_to_point_cloud(void *t, void *d, int cam, void *o) { (void)t; (void)d; (void)cam; (void)o; return 0; }

/* ---- k4arecord registry ---- */
#define MAX_REC 8
#define MAX_TAGS 16
#define MAX_TRACKS 8
#define MAX_BLOCKS 256
typedef struct { char name[64]; char value[256]; } tag_t;
typedef struct { char name[128]; uint8_t *data; size_t size; } attach_t;
typedef struct { char name[64]; char codec_id[64]; uint8_t *ctx; size_t ctx_size; uint64_t w, h, fps; int is_video; int is_sub; bool high_freq; } track_t;
typedef struct { uint64_t ts; uint8_t *data; size_t size; } block_t;
typedef struct {
    int in_use; int open; char path[512]; dev_cfg_t cfg; int has_device;
    tag_t tags[MAX_TAGS]; int tag_count;
    attach_t attach[8]; int attach_count;
    track_t tracks[MAX_TRACKS]; int track_count;
    int imu_track; int header_written;
    uint64_t cap_ts[1024]; int cap_count;
    imu_t imu[1024]; int imu_count;
    block_t blocks[MAX_TRACKS][MAX_BLOCKS]; int block_count[MAX_TRACKS];
} rec_t;
static rec_t g_recs[MAX_REC];
typedef struct { int in_use; int slot; int cap_pos; int imu_pos; int block_pos[MAX_TRACKS]; } play_t;
static play_t g_plays[64];
static rec_t *rec_of(void *h) {
    uintptr_t v = (uintptr_t)h;
    if (v < 0x7000 || v >= 0x7000 + MAX_REC) return NULL;
    rec_t *r = &g_recs[v - 0x7000];
    return (r->in_use && r->open) ? r : NULL;
}
static play_t *play_of(void *h) {
    uintptr_t v = (uintptr_t)h;
    if (v < 0x8000 || v >= 0x8000 + 64) return NULL;
    play_t *p = &g_plays[v - 0x8000];
    return p->in_use ? p : NULL;
}
static int track_slot(rec_t *r, const char *name) {
    for (int i = 0; i < r->track_count; i++)
        if (strcmp(r->tracks[i].name, name) == 0) return i;
    return -1;
}
static int builtin_enabled(rec_t *r, const char *name) {
    if (strcmp(name, "COLOR") == 0) return r->cfg.color_resolution != 0;
    if (strcmp(name, "DEPTH") == 0) return r->cfg.depth_mode != 0;
    if (strcmp(name, "IR") == 0) return r->cfg.depth_mode != 0;
    if (strcmp(name, "IMU") == 0) return r->imu_track;
    return 0;
}
static int track_list(rec_t *r, char names[][64]) {
    int n = 0;
    const char *builtins[4] = {"COLOR", "DEPTH", "IR", "IMU"};
    for (int i = 0; i < 4; i++)
        if (builtin_enabled(r, builtins[i])) { strcpy(names[n++], builtins[i]); }
    for (int i = 0; i < r->track_count; i++) strcpy(names[n++], r->tracks[i].name);
    return n;
}
int k4a_record_create(const char *path, void *dev, const dev_cfg_t cfg, void **handle) {
    int pick = -1;
    for (int i = 0; i < MAX_REC; i++)
        if (g_recs[i].in_use && strcmp(g_recs[i].path, path) == 0)
            { pick = i; break; }
    if (pick < 0)
        for (int i = 0; i < MAX_REC; i++)
            if (!g_recs[i].in_use) { pick = i; break; }
    if (pick < 0)
        for (int i = 0; i < MAX_REC; i++)
            if (g_recs[i].in_use && !g_recs[i].open)
                { pick = i; break; }
    if (pick < 0) return 1;
    {
        int i = pick;
        memset(&g_recs[i], 0, sizeof(rec_t));
        g_recs[i].in_use = 1; g_recs[i].open = 1;
        strncpy(g_recs[i].path, path, sizeof(g_recs[i].path) - 1);
        g_recs[i].cfg = cfg;
        g_recs[i].has_device = dev != NULL;
        FILE *f = fopen(path, "wb"); if (f) fclose(f);
        *handle = (void*)(uintptr_t)(0x7000 + i);
        return 0;
    }
}
int k4a_record_add_tag(void *h, const char *name, const char *value) {
    rec_t *r = rec_of(h); if (!r || r->tag_count >= MAX_TAGS) return 1;
    strncpy(r->tags[r->tag_count].name, name, 63);
    strncpy(r->tags[r->tag_count].value, value, 255);
    r->tag_count++; return 0;
}
int k4a_record_add_imu_track(void *h) {
    rec_t *r = rec_of(h); if (!r) return 1; r->imu_track = 1; return 0;
}
int k4a_record_add_attachment(void *h, const char *name, const uint8_t *buf, size_t sz) {
    rec_t *r = rec_of(h); if (!r || r->attach_count >= 8) return 1;
    attach_t *a = &r->attach[r->attach_count++];
    strncpy(a->name, name, 127);
    a->data = malloc(sz ? sz : 1); memcpy(a->data, buf, sz); a->size = sz;
    return 0;
}
int k4a_record_add_custom_video_track(void *h, const char *name, const char *codec, const uint8_t *ctx, size_t ctx_sz, const vid_settings_t *vs) {
    rec_t *r = rec_of(h); if (!r || r->track_count >= MAX_TRACKS) return 1;
    track_t *t = &r->tracks[r->track_count++];
    strncpy(t->name, name, 63); strncpy(t->codec_id, codec, 63);
    t->ctx = malloc(ctx_sz ? ctx_sz : 1); memcpy(t->ctx, ctx, ctx_sz); t->ctx_size = ctx_sz;
    t->w = vs->width; t->h = vs->height; t->fps = vs->frame_rate; t->is_video = 1;
    return 0;
}
int k4a_record_add_custom_subtitle_track(void *h, const char *name, const char *codec, const uint8_t *ctx, size_t ctx_sz, const sub_settings_t *ss) {
    rec_t *r = rec_of(h); if (!r || r->track_count >= MAX_TRACKS) return 1;
    track_t *t = &r->tracks[r->track_count++];
    strncpy(t->name, name, 63); strncpy(t->codec_id, codec, 63);
    t->ctx = malloc(ctx_sz ? ctx_sz : 1); memcpy(t->ctx, ctx, ctx_sz); t->ctx_size = ctx_sz;
    t->high_freq = ss ? ss->high_freq_data : false; t->is_sub = 1;
    return 0;
}
int k4a_record_write_header(void *h) {
    rec_t *r = rec_of(h); if (!r) return 1; r->header_written = 1; return 0;
}
static uint64_t cap_ts_of(void *cap) {
    uintptr_t v = (uintptr_t)cap;
    if (v >= 0x2000 && v < 0x2000 + 100000) return (uint64_t)(v - 0x2000) * 33333;
    return 0;
}
int k4a_record_write_capture(void *h, void *cap) {
    rec_t *r = rec_of(h);
    if (!r || !r->header_written || r->cap_count >= 1024) return 1;
    r->cap_ts[r->cap_count++] = cap_ts_of(cap);
    return 0;
}
int k4a_record_write_imu_sample(void *h, imu_t sample) {
    rec_t *r = rec_of(h);
    if (!r || !r->header_written || r->imu_count >= 1024) return 1;
    r->imu[r->imu_count++] = sample;
    return 0;
}
int k4a_record_write_custom_track_data(void *h, const char *name, uint64_t ts, uint8_t *data, size_t sz) {
    rec_t *r = rec_of(h); if (!r || !r->header_written) return 1;
    int slot = track_slot(r, name); if (slot < 0) return 1;
    if (r->block_count[slot] >= MAX_BLOCKS) return 1;
    block_t *b = &r->blocks[slot][r->block_count[slot]++];
    b->ts = ts; b->data = malloc(sz ? sz : 1); memcpy(b->data, data, sz); b->size = sz;
    return 0;
}
int k4a_record_flush(void *h) { return rec_of(h) ? 0 : 1; }
void k4a_record_close(void *h) {
    uintptr_t v = (uintptr_t)h;
    if (v < 0x7000 || v >= 0x7000 + MAX_REC) return;
    g_recs[v - 0x7000].open = 0;
}
int k4a_playback_open(const char *path, void **handle) {
    for (int i = 0; i < MAX_REC; i++) {
        if (!g_recs[i].in_use || strcmp(g_recs[i].path, path) != 0) continue;
        for (int j = 0; j < 64; j++) {
            if (g_plays[j].in_use) continue;
            play_t *p = &g_plays[j];
            memset(p, 0, sizeof(*p)); p->in_use = 1; p->slot = i;
            *handle = (void*)(uintptr_t)(0x8000 + j);
            return 0;
        }
        return 1;
    }
    return 1;
}
int k4a_playback_get_raw_calibration(void *h, uint8_t *data, size_t *size) {
    (void)h; (void)data; (void)size; return 1;
}
int k4a_playback_get_calibration(void *h, cal_t *cal) {
    play_t *p = play_of(h); if (!p) return 1;
    rec_t *r = &g_recs[p->slot];
    memset(cal, 0, sizeof(*cal));
    cal->depth_mode = r->cfg.depth_mode; cal->color_resolution = r->cfg.color_resolution;
    return 0;
}
int k4a_playback_get_record_configuration(void *h, rec_cfg_t *cfg) {
    play_t *p = play_of(h); if (!p) return 1;
    rec_t *r = &g_recs[p->slot];
    memset(cfg, 0, sizeof(*cfg));
    cfg->color_format = r->cfg.color_format;
    cfg->color_resolution = r->cfg.color_resolution;
    cfg->depth_mode = r->cfg.depth_mode;
    cfg->camera_fps = r->cfg.camera_fps;
    cfg->color_track_enabled = r->cfg.color_resolution != 0;
    cfg->depth_track_enabled = r->cfg.depth_mode != 0;
    cfg->ir_track_enabled = r->cfg.depth_mode != 0;
    cfg->imu_track_enabled = r->imu_track;
    cfg->depth_delay_off_color_usec = r->cfg.depth_delay_off_color_usec;
    cfg->wired_sync_mode = r->cfg.wired_sync_mode;
    cfg->subordinate_delay_off_master_usec = r->cfg.subordinate_delay_off_master_usec;
    cfg->start_timestamp_offset_usec = 0;
    return 0;
}
bool k4a_playback_check_track_exists(void *h, const char *name) {
    play_t *p = play_of(h); if (!p) return false;
    rec_t *r = &g_recs[p->slot];
    return builtin_enabled(r, name) || track_slot(r, name) >= 0;
}
size_t k4a_playback_get_track_count(void *h) {
    play_t *p = play_of(h); if (!p) return 0;
    char names[16][64]; return (size_t)track_list(&g_recs[p->slot], names);
}
int k4a_playback_get_track_name(void *h, size_t idx, char *name, size_t *size) {
    play_t *p = play_of(h); if (!p) return 1;
    char names[16][64]; int n = track_list(&g_recs[p->slot], names);
    if ((int)idx >= n) return 1;
    size_t need = strlen(names[idx]) + 1;
    if (!name || *size < need) { *size = need; return 2; }
    memcpy(name, names[idx], need); *size = need; return 0;
}
bool k4a_playback_track_is_builtin(void *h, const char *name) {
    play_t *p = play_of(h); if (!p) return false;
    return strcmp(name, "COLOR") == 0 || strcmp(name, "DEPTH") == 0 ||
           strcmp(name, "IR") == 0 || strcmp(name, "IMU") == 0;
}
int k4a_playback_track_get_video_settings(void *h, const char *name, vid_settings_t *vs) {
    play_t *p = play_of(h); if (!p) return 1;
    rec_t *r = &g_recs[p->slot];
    int slot = track_slot(r, name);
    if (slot < 0 || !r->tracks[slot].is_video) return 1;
    vs->width = r->tracks[slot].w; vs->height = r->tracks[slot].h; vs->frame_rate = r->tracks[slot].fps;
    return 0;
}
static int read_string(const char *src, char *dst, size_t *size) {
    size_t need = strlen(src) + 1;
    if (!dst || *size < need) { *size = need; return 2; }
    memcpy(dst, src, need); *size = need; return 0;
}
static int read_bytes(const uint8_t *src, size_t src_sz, uint8_t *dst, size_t *size) {
    if (!dst || *size < src_sz) { *size = src_sz; return 2; }
    memcpy(dst, src, src_sz); *size = src_sz; return 0;
}
int k4a_playback_track_get_codec_id(void *h, const char *name, char *id, size_t *size) {
    play_t *p = play_of(h); if (!p) return 1;
    rec_t *r = &g_recs[p->slot];
    int slot = track_slot(r, name); if (slot < 0) return 1;
    return read_string(r->tracks[slot].codec_id, id, size);
}
int k4a_playback_track_get_codec_context(void *h, const char *name, uint8_t *ctx, size_t *size) {
    play_t *p = play_of(h); if (!p) return 1;
    rec_t *r = &g_recs[p->slot];
    int slot = track_slot(r, name); if (slot < 0) return 1;
    return read_bytes(r->tracks[slot].ctx, r->tracks[slot].ctx_size, ctx, size);
}
int k4a_playback_get_tag(void *h, const char *name, char *val, size_t *size) {
    play_t *p = play_of(h); if (!p) return 1;
    rec_t *r = &g_recs[p->slot];
    for (int i = 0; i < r->tag_count; i++)
        if (strcmp(r->tags[i].name, name) == 0) return read_string(r->tags[i].value, val, size);
    return 1;
}
int k4a_playback_set_color_conversion(void *h, int fmt) { (void)h; (void)fmt; return 0; }
int k4a_playback_get_attachment(void *h, const char *name, uint8_t *data, size_t *size) {
    play_t *p = play_of(h); if (!p) return 1;
    rec_t *r = &g_recs[p->slot];
    for (int i = 0; i < r->attach_count; i++)
        if (strcmp(r->attach[i].name, name) == 0) return read_bytes(r->attach[i].data, r->attach[i].size, data, size);
    return 1;
}
int k4a_playback_get_next_capture(void *h, void **cap) {
    play_t *p = play_of(h); if (!p) return 1;
    rec_t *r = &g_recs[p->slot];
    if (p->cap_pos >= r->cap_count) return 2;
    uint64_t ts = r->cap_ts[p->cap_pos++];
    g_cur_ts = ts;
    *cap = (void*)(uintptr_t)(0x2000 + (ts / 33333));
    return 0;
}
int k4a_playback_get_previous_capture(void *h, void **cap) {
    play_t *p = play_of(h); if (!p) return 1;
    rec_t *r = &g_recs[p->slot];
    if (p->cap_pos <= 0) return 2;
    uint64_t ts = r->cap_ts[--p->cap_pos];
    g_cur_ts = ts;
    *cap = (void*)(uintptr_t)(0x2000 + (ts / 33333));
    return 0;
}
int k4a_playback_get_next_imu_sample(void *h, imu_t *s) {
    play_t *p = play_of(h); if (!p) return 1;
    rec_t *r = &g_recs[p->slot];
    if (p->imu_pos >= r->imu_count) return 2;
    *s = r->imu[p->imu_pos++];
    return 0;
}
int k4a_playback_get_previous_imu_sample(void *h, imu_t *s) {
    play_t *p = play_of(h); if (!p) return 1;
    rec_t *r = &g_recs[p->slot];
    if (p->imu_pos <= 0) return 2;
    *s = r->imu[--p->imu_pos];
    return 0;
}
static int data_block_at(play_t *p, rec_t *r, const char *name, int next, void **block) {
    int slot = track_slot(r, name);
    if (slot < 0) return 1;
    if (next) {
        if (p->block_pos[slot] >= r->block_count[slot]) return 2;
        int idx = p->block_pos[slot]++;
        *block = (void*)(uintptr_t)(0xA000 + slot * 1024 + idx);
        return 0;
    }
    if (p->block_pos[slot] <= 0) return 2;
    int idx = --p->block_pos[slot];
    *block = (void*)(uintptr_t)(0xA000 + slot * 1024 + idx);
    return 0;
}
int k4a_playback_get_next_data_block(void *h, const char *name, void **block) {
    play_t *p = play_of(h); if (!p) return 1;
    return data_block_at(p, &g_recs[p->slot], name, 1, block);
}
int k4a_playback_get_previous_data_block(void *h, const char *name, void **block) {
    play_t *p = play_of(h); if (!p) return 1;
    return data_block_at(p, &g_recs[p->slot], name, 0, block);
}
static block_t *block_of(void *h) {
    uintptr_t v = (uintptr_t)h;
    if (v < 0xA000 || v >= 0xA000 + 8 * 1024) return NULL;
    int slot = (int)(v - 0xA000) / 1024, idx = (int)(v - 0xA000) % 1024;
    /* find the recording that owns this block: only one playback is
       live in tests at a time; scan the registry for a matching slot */
    for (int i = 0; i < MAX_REC; i++) {
        if (!g_recs[i].in_use) continue;
        if (idx < g_recs[i].block_count[slot]) return &g_recs[i].blocks[slot][idx];
    }
    return NULL;
}
uint64_t k4a_playback_data_block_get_device_timestamp_usec(void *h) {
    block_t *b = block_of(h); return b ? b->ts : 0;
}
size_t k4a_playback_data_block_get_buffer_size(void *h) {
    block_t *b = block_of(h); return b ? b->size : 0;
}
uint8_t *k4a_playback_data_block_get_buffer(void *h) {
    block_t *b = block_of(h); return b ? b->data : NULL;
}
void k4a_playback_data_block_release(void *h) { (void)h; }
int k4a_playback_seek_timestamp(void *h, int64_t offset, int origin) {
    play_t *p = play_of(h); if (!p) return 1;
    rec_t *r = &g_recs[p->slot];
    uint64_t target;
    if (origin == 1) { /* END: offset is relative to the end */
        uint64_t len = r->cap_count ? r->cap_ts[r->cap_count - 1] - r->cap_ts[0] : 0;
        int64_t t = (int64_t)len + offset;
        target = t < 0 ? 0 : (uint64_t)t;
    } else {
        target = offset < 0 ? 0 : (uint64_t)offset;
    }
    int pos = 0;
    while (pos < r->cap_count && r->cap_ts[pos] < target) pos++;
    p->cap_pos = pos;
    int ipos = 0;
    while (ipos < r->imu_count && r->imu[ipos].acc_timestamp_usec < target) ipos++;
    p->imu_pos = ipos;
    for (int s = 0; s < r->track_count; s++) {
        int bpos = 0;
        while (bpos < r->block_count[s] && r->blocks[s][bpos].ts < target) bpos++;
        p->block_pos[s] = bpos;
    }
    return 0;
}
uint64_t k4a_playback_get_recording_length_usec(void *h) {
    play_t *p = play_of(h); if (!p) return 0;
    rec_t *r = &g_recs[p->slot];
    if (r->cap_count < 2) return 0;
    return r->cap_ts[r->cap_count - 1] - r->cap_ts[0];
}
uint64_t k4a_playback_get_last_timestamp_usec(void *h) {
    play_t *p = play_of(h); if (!p) return 0;
    rec_t *r = &g_recs[p->slot];
    return r->cap_count ? r->cap_ts[r->cap_count - 1] : 0;
}
void k4a_playback_close(void *h) {
    play_t *p = play_of(h);
    if (p) p->in_use = 0;
}
"""

LAYOUT_C = r"""
#include <stdio.h>
#include <stddef.h>
#include <k4a/k4a.h>
#include <k4arecord/types.h>
int main(void) {
    printf("RecordConfiguration %zu\n", sizeof(k4a_record_configuration_t));
    printf("RecordConfiguration.color_format %zu\n", offsetof(k4a_record_configuration_t, color_format));
    printf("RecordConfiguration.color_resolution %zu\n", offsetof(k4a_record_configuration_t, color_resolution));
    printf("RecordConfiguration.depth_mode %zu\n", offsetof(k4a_record_configuration_t, depth_mode));
    printf("RecordConfiguration.camera_fps %zu\n", offsetof(k4a_record_configuration_t, camera_fps));
    printf("RecordConfiguration.color_track_enabled %zu\n", offsetof(k4a_record_configuration_t, color_track_enabled));
    printf("RecordConfiguration.depth_track_enabled %zu\n", offsetof(k4a_record_configuration_t, depth_track_enabled));
    printf("RecordConfiguration.ir_track_enabled %zu\n", offsetof(k4a_record_configuration_t, ir_track_enabled));
    printf("RecordConfiguration.imu_track_enabled %zu\n", offsetof(k4a_record_configuration_t, imu_track_enabled));
    printf("RecordConfiguration.depth_delay_off_color_usec %zu\n", offsetof(k4a_record_configuration_t, depth_delay_off_color_usec));
    printf("RecordConfiguration.wired_sync_mode %zu\n", offsetof(k4a_record_configuration_t, wired_sync_mode));
    printf("RecordConfiguration.subordinate_delay_off_master_usec %zu\n", offsetof(k4a_record_configuration_t, subordinate_delay_off_master_usec));
    printf("RecordConfiguration.start_timestamp_offset_usec %zu\n", offsetof(k4a_record_configuration_t, start_timestamp_offset_usec));
    printf("RecordVideoSettings %zu\n", sizeof(k4a_record_video_settings_t));
    printf("RecordVideoSettings.width %zu\n", offsetof(k4a_record_video_settings_t, width));
    printf("RecordVideoSettings.height %zu\n", offsetof(k4a_record_video_settings_t, height));
    printf("RecordVideoSettings.frame_rate %zu\n", offsetof(k4a_record_video_settings_t, frame_rate));
    printf("RecordSubtitleSettings %zu\n", sizeof(k4a_record_subtitle_settings_t));
    printf("RecordSubtitleSettings.high_freq_data %zu\n", offsetof(k4a_record_subtitle_settings_t, high_freq_data));
    printf("Calibration %zu\n", sizeof(k4a_calibration_t));
    printf("Calibration.depth_mode %zu\n", offsetof(k4a_calibration_t, depth_mode));
    printf("Calibration.color_resolution %zu\n", offsetof(k4a_calibration_t, color_resolution));
    return 0;
}
"""


def _compile(source: str, out: Path, *, shared: bool) -> None:
    src = out.with_suffix(".c")
    src.write_text(source, encoding="utf-8")
    cmd = ["gcc", "-O1", "-fPIC", "-Wall", "-Werror"]
    if shared:
        cmd.append("-shared")
    cmd += [f"-I{VENDOR_INCLUDE}", str(src), "-o", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"gcc failed: {proc.stderr}"


@pytest.fixture(scope="session")
def fake_lib(tmp_path_factory):
    """The fake libk4a+libk4arecord .so, compiled once per session."""
    if subprocess.run(["which", "gcc"], capture_output=True).returncode != 0:
        pytest.skip("gcc not available — cannot build the fake SDK")
    out = tmp_path_factory.mktemp("fakek4a") / "libfake.so"
    _compile(FAKE_C, out, shared=True)
    return out


@pytest.fixture()
def fake_env(fake_lib, monkeypatch):
    """Point both library loaders at the fake .so."""
    monkeypatch.setenv("K4A_LIB", str(fake_lib))
    monkeypatch.setenv("K4A_RECORD_LIB", str(fake_lib))
    return fake_lib


# -- pure shape ------------------------------------------------------


def test_stream_result_values():
    assert int(krec.StreamResult.SUCCEEDED) == 0
    assert int(krec.StreamResult.FAILED) == 1
    assert int(krec.StreamResult.EOF) == 2


def test_seek_origin_values():
    assert int(krec.PlaybackSeekOrigin.BEGIN) == 0
    assert int(krec.PlaybackSeekOrigin.END) == 1
    assert int(krec.PlaybackSeekOrigin.DEVICE_TIME) == 2


def test_track_name_constants():
    assert krec.TRACK_COLOR == "COLOR"
    assert krec.TRACK_DEPTH == "DEPTH"
    assert krec.TRACK_IR == "IR"
    assert krec.TRACK_IMU == "IMU"


def test_record_struct_layout_matches_gcc(tmp_path):
    """ctypes struct sizes/offsets vs gcc's reading of the vendored headers."""
    exe = tmp_path / "layout"
    _compile(LAYOUT_C, exe, shared=False)
    proc = subprocess.run([str(exe)], capture_output=True, text=True,
                          timeout=30)
    assert proc.returncode == 0, proc.stderr
    got: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        name, _, value = line.partition(" ")
        got[name] = int(value)
    structs = {
        "RecordConfiguration": krec.RecordConfiguration,
        "RecordVideoSettings": krec.RecordVideoSettings,
        "RecordSubtitleSettings": krec.RecordSubtitleSettings,
    }
    for name, cls in structs.items():
        assert ctypes.sizeof(cls) == got[name], name
        for field, _type in cls._fields_:
            key = f"{name}.{field}"
            assert int(getattr(cls, field).offset) == got[key], key
    from nomorals.hardware.kinect.bindings import Calibration as CCal
    assert ctypes.sizeof(CCal) == got["Calibration"]
    assert int(CCal.depth_mode.offset) == got["Calibration.depth_mode"]
    assert int(CCal.color_resolution.offset) == got[
        "Calibration.color_resolution"]


def test_record_configuration_field_order():
    names = [name for name, _ in krec.RecordConfiguration._fields_]
    assert names == [
        "color_format", "color_resolution", "depth_mode", "camera_fps",
        "color_track_enabled", "depth_track_enabled", "ir_track_enabled",
        "imu_track_enabled", "depth_delay_off_color_usec",
        "wired_sync_mode", "subordinate_delay_off_master_usec",
        "start_timestamp_offset_usec",
    ]


def test_tag_name_validation():
    for good in ("LOCATION", "CAMERA-1", "TAKE_2", "A"):
        krec.validate_tag_name(good)
    for bad in ("lower", "has space", "has.dot", "", "ü"):
        with pytest.raises(krec.RecordError):
            krec.validate_tag_name(bad)


def test_load_record_library_missing_raises(monkeypatch):
    monkeypatch.setenv("K4A_RECORD_LIB", "/nonexistent/libk4arecord.so")
    with pytest.raises(NoSdkError, match="libk4arecord"):
        krec.load_record_library()
    monkeypatch.delenv("K4A_RECORD_LIB")
    monkeypatch.setattr(krec, "RECORD_SONAMES", ("libk4arecord-nope.so",))
    with pytest.raises(NoSdkError, match="tried"):
        krec.load_record_library()


def test_recorder_without_library_raises(monkeypatch):
    monkeypatch.setenv("K4A_RECORD_LIB", "/nonexistent/libk4arecord.so")
    with pytest.raises(NoSdkError, match="libk4arecord"):
        krec.Recorder.create("/tmp/x.mkv", DeviceConfig())


def test_playback_without_library_raises(monkeypatch):
    monkeypatch.setenv("K4A_RECORD_LIB", "/nonexistent/libk4arecord.so")
    with pytest.raises(NoSdkError, match="libk4arecord"):
        krec.Playback.open("/tmp/x.mkv")


# -- fake-library sessions -------------------------------------------


def test_fake_library_binds_every_symbol(fake_env):
    from nomorals.hardware.kinect.bindings import load_library
    k4a = load_library()
    assert len(k4a.functions) == 65
    record = krec.load_record_library()
    assert len(record.functions) == 40
    assert callable(record.k4a_record_create)
    assert "k4a_record_create" in record.functions
    assert "k4a_playback_open" in record.functions
    assert "k4a_playback_seek_timestamp" in record.functions


def test_device_handle_and_raw_capture(fake_env):
    with KinectDevice.open(0) as device:
        assert device.handle == 0x1001
        assert device.serial_number() == "FAKE123456"
        assert device.version()["rgb"] == {
            "major": 1, "minor": 6, "iteration": 5}
        with device.raw_capture() as handle:
            assert handle is not None and handle >= 0x2000
        with device.raw_capture(timeout_ms=5) as handle:
            assert handle is not None


def test_record_then_playback_roundtrip(fake_env, tmp_path):
    path = tmp_path / "session.mkv"
    summary = krec.record_live(
        str(path), frames=5, tags={"LOCATION": "TEST-RIG"},
        imu=True)
    assert summary["frames_written"] == 5
    assert summary["imu_samples"] == 5
    assert summary["path"] == str(path)
    assert path.exists()

    with krec.Playback.open(str(path)) as playback:
        config = playback.record_configuration()
        assert config["color_resolution"] == "RES_720P"
        assert config["depth_mode"] == "WFOV_2X2BINNED"
        assert config["color_track_enabled"] is True
        assert config["imu_track_enabled"] is True
        assert playback.track_count() == 4
        assert playback.track_names() == ["COLOR", "DEPTH", "IR", "IMU"]
        assert playback.track_exists("COLOR") is True
        assert playback.track_exists("NOPE") is False
        assert playback.track_is_builtin("DEPTH") is True
        assert playback.get_tag("LOCATION") == "TEST-RIG"
        assert playback.get_tag("MISSING") is None
        # The fake's device clock is global across the process, so
        # timestamps are asserted relatively, never absolutely.
        assert playback.recording_length_usec() == 4 * 33333

        captures = []
        while True:
            capture = playback.next_capture()
            if capture is None:
                break
            captures.append(capture)
        assert len(captures) == 5
        stamps = [c.depth.device_timestamp_usec for c in captures]
        assert stamps == [stamps[0] + i * 33333 for i in range(5)]
        assert playback.last_timestamp_usec() == stamps[-1]
        first = captures[0]
        assert first.has_color and first.has_depth
        assert first.ir is not None
        assert first.color.width == 1280 and first.color.height == 720
        assert first.depth.width == 640 and first.depth.height == 576
        assert first.depth.depth_values()[0][0] == 1000
        assert first.temperature_c == 25.0
        assert playback.next_capture() is None  # EOF stays EOF

        back = playback.previous_capture()
        assert back is not None
        assert back.depth.depth_values()[0][0] == 1000

        samples = []
        while True:
            sample = playback.next_imu_sample()
            if sample is None:
                break
            samples.append(sample)
        assert len(samples) == 5
        assert samples[0]["temperature_c"] == 36.5
        assert samples[0]["acc"] == pytest.approx((0.0, 0.0, 9.81))

        playback.seek(stamps[2], krec.PlaybackSeekOrigin.BEGIN)
        capture = playback.next_capture()
        assert capture is not None
        assert capture.depth.device_timestamp_usec == stamps[2]


def test_recorder_manual_session(fake_env, tmp_path):
    path = tmp_path / "manual.mkv"
    config = DeviceConfig()
    with krec.Recorder.create(str(path), config,
                              tags={"TAKE": "7"}) as recorder:
        recorder.add_imu_track()
        recorder.add_attachment("notes.txt", b"fake session")
        recorder.add_custom_video_track(
            "OVERLAY", "V_MPEG4/ISO/AVC", b"\x00\x01",
            width=320, height=240, frame_rate=30)
        recorder.add_custom_subtitle_track(
            "EVENTS", "S_TEXT/UTF8", b"", high_freq_data=True)
        recorder.write_header()
        with KinectDevice.open(0) as device:
            with device.raw_capture() as handle:
                assert handle is not None
                recorder.write_capture_handle(handle)
            recorder.write_imu_sample({
                "temperature_c": 36.5,
                "acc": (0.0, 0.0, 9.81),
                "acc_timestamp_usec": 1000,
                "gyro": (0.1, 0.2, 0.3),
                "gyro_timestamp_usec": 1000})
        recorder.write_custom_track_data("OVERLAY", 1000, b"\xde\xad")
        recorder.flush()
    with krec.Playback.open(str(path)) as playback:
        assert playback.get_tag("TAKE") == "7"
        assert playback.get_attachment("notes.txt") == b"fake session"
        assert playback.track_exists("OVERLAY") is True
        assert playback.track_is_builtin("OVERLAY") is False
        settings = playback.track_video_settings("OVERLAY")
        assert settings == {"width": 320, "height": 240,
                            "frame_rate": 30}
        assert playback.track_codec_id("OVERLAY") == "V_MPEG4/ISO/AVC"
        assert playback.track_codec_context("OVERLAY") == b"\x00\x01"
        block = playback.next_data_block("OVERLAY")
        assert block is not None
        assert block.device_timestamp_usec == 1000
        assert block.data == b"\xde\xad"
        assert playback.next_data_block("OVERLAY") is None
        assert playback.get_tag("NOPE") is None


def test_recorder_write_before_header_refused(fake_env, tmp_path):
    path = tmp_path / "early.mkv"
    with krec.Recorder.create(str(path), DeviceConfig()) as recorder, \
            KinectDevice.open(0) as device, \
            device.raw_capture() as handle, \
            pytest.raises(krec.RecordError):
        recorder.write_capture_handle(handle)


def test_recorder_writes_decoded_capture(fake_env, tmp_path):
    """A Capture dataclass (e.g. from the vendored fixture) records
    without a live device handle."""
    path = tmp_path / "dataclass.mkv"
    raw = (FIXTURES / "depth_nfov_640x576.raw").read_bytes()
    frame = Frame(fmt=ImageFormat.DEPTH16, width=640, height=576,
                  stride=1280, data=raw,
                  device_timestamp_usec=4242)
    capture = Capture(depth=frame, temperature_c=24.0)
    with krec.Recorder.create(str(path), DeviceConfig()) as recorder:
        recorder.write_header()
        recorder.write_capture(capture)
    with krec.Playback.open(str(path)) as playback:
        # COLOR + DEPTH + IR are enabled by the default config;
        # no IMU track was added, so there are exactly 3 tracks.
        assert playback.track_count() == 3
        got = playback.next_capture()
        assert got is not None
        assert got.has_depth


def test_playback_open_unknown_path_fails(fake_env, tmp_path):
    missing = tmp_path / "nope.mkv"
    with pytest.raises(FileNotFoundError):
        krec.Playback.open(str(missing))
    real_but_unknown = tmp_path / "empty.mkv"
    real_but_unknown.write_bytes(b"")
    with pytest.raises(krec.PlaybackError):
        krec.Playback.open(str(real_but_unknown))


def test_playback_raw_calibration_absent_is_none(fake_env, tmp_path):
    path = tmp_path / "nocal.mkv"
    krec.record_live(str(path), frames=1, imu=False)
    with krec.Playback.open(str(path)) as playback:
        assert playback.raw_calibration() is None
        calibration = playback.calibration()
        assert calibration.depth_mode == DepthMode.WFOV_2X2BINNED
        assert calibration.color_resolution == ColorResolution.RES_720P


def test_probe_reports_record_library(fake_env):
    from nomorals.hardware.kinect.probe import probe as probe_fn
    report = probe_fn()
    assert report.sdk_present is True
    assert report.record_sdk_present is True
    assert report.record_sdk_path is not None
    assert report.device_count == 1


def test_probe_without_record_library():
    from nomorals.hardware.kinect.probe import probe as probe_fn
    report = probe_fn()
    assert report.record_sdk_present is False
    assert report.record_sdk_error is not None


# -- tool wiring -----------------------------------------------------


def _mocap_call(**kwargs):
    """Call the mocap brain tool through the registry, like the
    mocap-state tests do. Returns the outcome (ok / value / error)."""
    from types import SimpleNamespace

    from nomorals.tools.registry import ToolRegistry
    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    return registry.call("mocap", **kwargs)


def _mocap_ok(**kwargs):
    outcome = _mocap_call(**kwargs)
    assert getattr(outcome, "ok", False), \
        f"mocap call failed: {getattr(outcome, 'error', outcome)}"
    return outcome.value


def test_mocap_tool_lists_kinect_actions():
    from nomorals.tools import mocap as mocap_tool
    assert "kinect_record" in mocap_tool._ACTIONS
    assert "kinect_playback" in mocap_tool._ACTIONS


def test_mocap_tool_kinect_record_and_playback(fake_env, tmp_path,
                                               monkeypatch):
    out = tmp_path / "tool.mkv"
    result = _mocap_ok(action="kinect_record", output=str(out),
                       params={"frames": 3, "imu": True,
                               "tags": {"LOCATION": "TOOL"}})
    assert result["frames_written"] == 3
    assert result["imu_samples"] == 3
    info = _mocap_ok(action="kinect_playback", sources=[str(out)],
                     params={"op": "info"})
    assert info["track_count"] == 4
    assert info["track_names"] == ["COLOR", "DEPTH", "IR", "IMU"]
    read = _mocap_ok(action="kinect_playback", sources=[str(out)],
                     params={"op": "read", "count": 2})
    assert read["captures_returned"] == 2
    assert read["captures"][0]["depth"]["width"] == 640
    tag = _mocap_ok(action="kinect_playback", sources=[str(out)],
                    params={"op": "tag", "name": "LOCATION"})
    assert tag["value"] == "TOOL"


def test_mocap_tool_kinect_record_without_library(monkeypatch):
    monkeypatch.setenv("K4A_RECORD_LIB", "/nonexistent/libk4arecord.so")
    outcome = _mocap_call(action="kinect_record", output="/tmp/x.mkv",
                          params={"frames": 1})
    assert not getattr(outcome, "ok", False)
    assert "libk4arecord" in str(getattr(outcome, "error", ""))


def test_mocap_tool_kinect_status_reports_record_sdk(fake_env):
    status = _mocap_ok(action="kinect_status")
    assert status["sdk_present"] is True
    assert status["record_sdk_present"] is True


def test_attribution_echelon_section_verified():
    from nomorals.hardware.kinect.attribution import verify_attribution
    result = verify_attribution()
    assert result["failed"] == [], result["failed"]
    assert result["verified"] > 30
    assert result["skipped"] == []


def test_kinect_package_exports_record_names():
    from nomorals.hardware import kinect
    for name in ("Recorder", "Playback", "RecordConfiguration",
                 "StreamResult", "PlaybackSeekOrigin",
                 "load_record_library", "record_live",
                 "RecordError", "PlaybackError"):
        assert hasattr(kinect, name), name


def test_record_module_attribution_names_upstream():
    text = (REPO / "nomorals" / "hardware" / "kinect" / "record.py"
            ).read_text(encoding="utf-8")
    assert "Amédée" in text
    assert "MIT" in text
    assert "echelon/k4a-sys" in text


# ── CLI: nm mocap kinect ─────────────────────────────────────────────


def test_cli_kinect_status_without_library(monkeypatch):
    from nomorals.cmdline.dispatch import main
    monkeypatch.setenv("K4A_LIB", "/nonexistent/libk4a.so")
    monkeypatch.setenv("K4A_RECORD_LIB", "/nonexistent/libk4arecord.so")
    assert main(["mocap", "kinect", "status"]) == 0


def test_cli_kinect_record_and_playback(fake_env, tmp_path, capsys):
    from nomorals.cmdline.dispatch import main
    out = tmp_path / "cli.mkv"
    assert main(["mocap", "--json", "kinect", "record", str(out),
                 "--frames", "3", "--tag", "LOCATION=CLI"]) == 0
    text = capsys.readouterr().out
    assert json.loads(text)["frames_written"] == 3
    assert main(["mocap", "--json", "kinect", "playback", str(out),
                 "info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["configuration"]["color_resolution"] == "RES_720P"
    assert main(["mocap", "--json", "kinect", "playback", str(out),
                 "tag", "--name", "LOCATION"]) == 0
    tag = json.loads(capsys.readouterr().out)
    assert tag["value"] == "CLI"
    assert main(["mocap", "--json", "kinect", "playback", str(out),
                 "read", "--count", "2"]) == 0
    read = json.loads(capsys.readouterr().out)
    assert read["captures_returned"] == 2
    assert read["captures"][0]["depth"]["width"] == 640


def test_cli_kinect_playback_bad_op(fake_env, tmp_path):
    from nomorals.cmdline.dispatch import main
    out = tmp_path / "bad.mkv"
    assert main(["mocap", "kinect", "record", str(out), "--frames",
                 "1"]) == 0
    # argparse rejects an op outside the choices (rc 2).
    assert main(["mocap", "kinect", "playback", str(out),
                 "bogus"]) == 2


def test_cli_kinect_record_bad_format(fake_env, tmp_path, capsys):
    from nomorals.cmdline.dispatch import main
    out = tmp_path / "badfmt.mkv"
    assert main(["mocap", "kinect", "record", str(out), "--frames",
                 "1", "--color-format", "NOPE"]) == 2
    assert "unknown color_format" in capsys.readouterr().err


# ── chat: /mocap kinect ─────────────────────────────────────────────


def test_chat_kinect_record_and_playback(fake_env, tmp_path):
    from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin
    mixin = RuntimeMocapMixin()
    out = tmp_path / "chat.mkv"
    reply = mixin._control_mocap(
        f"kinect record {out} frames=3 tag=LOCATION=CHAT", "test")
    assert "recorded 3 frames" in reply
    assert str(out) in reply
    info = mixin._control_mocap(f"kinect playback {out} info", "test")
    assert json.loads(info)["configuration"]["color_resolution"] \
        == "RES_720P"
    read = mixin._control_mocap(
        f"kinect playback {out} read count=2", "test")
    assert "read 2 captures" in read
    assert "@" in read and "color 1280x720" in read
    tracks = mixin._control_mocap(f"kinect playback {out} tracks",
                                  "test")
    assert "COLOR (builtin)" in tracks
    tag = mixin._control_mocap(
        f"kinect playback {out} tag name=LOCATION", "test")
    assert tag == "LOCATION: CHAT"


def test_chat_kinect_status_without_library(monkeypatch):
    from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin
    monkeypatch.setenv("K4A_LIB", "/nonexistent/libk4a.so")
    monkeypatch.setenv("K4A_RECORD_LIB", "/nonexistent/libk4arecord.so")
    mixin = RuntimeMocapMixin()
    status = mixin._control_mocap("kinect status", "test")
    assert "No Kinect record library either:" in status


def test_chat_kinect_record_without_library(monkeypatch, tmp_path):
    from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin
    monkeypatch.setenv("K4A_LIB", "/nonexistent/libk4a.so")
    monkeypatch.setenv("K4A_RECORD_LIB", "/nonexistent/libk4arecord.so")
    mixin = RuntimeMocapMixin()
    reply = mixin._control_mocap(
        f"kinect record {tmp_path}/nope.mkv frames=1", "test")
    assert reply.startswith("kinect: ")
    assert "libk4arecord" in reply


def test_chat_kinect_help_lists_verbs():
    from nomorals.agents.partner.runtime_mocap import RuntimeMocapMixin
    mixin = RuntimeMocapMixin()
    help_text = mixin._control_mocap("", "test")
    assert "kinect record" in help_text
    assert "kinect playback" in help_text
    assert "kinect status" in help_text


def test_control_mocap_help_names_kinect():
    from nomorals.social.chat.control import COMMAND_DETAILS
    usage = COMMAND_DETAILS["mocap"]["usage"]
    assert "kinect record" in usage
    assert "kinect playback" in usage
    assert "kinect status" in usage
