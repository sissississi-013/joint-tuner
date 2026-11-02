"""Render the tuning race in MuJoCo. A ControlEval 2nd-order plant num/(s^2+a s+b) is built exactly as a hinge joint:
inertia 1, joint damping a, joint stiffness b, motor gear num. Each tuner's logged step tests (same gains as the judge saw)
drive the joint through a PID with the benchmark's derivative filter. Usage: python demo/mujoco_race.py UID OUT.mp4"""
import sys, json, numpy as np, mujoco, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "control")
from judge import load_tasks, evaluate
UID, OUT = sys.argv[1], sys.argv[2]
T = {t["uid"]: t for t in load_tasks()}[UID]
num, den = np.array(T["num"], float), np.array(T["den"], float); num, den = num / den[0], den / den[0]
assert len(den) == 3 and len(num) == 1, "second-order plants only"
A, B, K = den[1], den[2], num[0]
row = next(r for r in json.load(open("runs/trace_v3.json"))["rows"] if r["uid"] == UID)
gpt = next(r for r in json.load(open("runs/pid_closed_gpt-6.1-terra_10.json")) if r["uid"] == UID)
gg = gpt["gains"]; gpt_g = [float(gg["Kp"]), float(gg["Ki"]), float(gg["Kd"])]
SIMT = 2.5 * T["settling_time_max"]; DT = 0.0005; SPEED = 4.0; FPS = 30; W, H = 480, 400; TARGET = 1.0  # rad
COL = {"cold": "0.45 0.50 0.55 1", "gpt": "0.36 0.45 0.90 1", "ours": "1.0 0.52 0.20 1"}
def xml(rgba):
    return f"""<mujoco><option timestep="{DT}" gravity="0 0 0" integrator="RK4"/>
<visual><global offwidth="{W}" offheight="{H}"/><quality shadowsize="2048"/></visual>
<asset><texture name="g" type="2d" builtin="checker" rgb1=".86 .88 .9" rgb2=".80 .82 .85" width="256" height="256"/>
<material name="g" texture="g" texrepeat="8 8"/><texture type="skybox" builtin="gradient" rgb1=".93 .95 .97" rgb2=".78 .82 .87" width="64" height="64"/></asset>
<worldbody><light pos="0 -1 3" dir="0 .3 -1" diffuse=".9 .9 .9"/>
<geom type="plane" size="2 2 .1" material="g"/>
<geom type="cylinder" pos="0 0 .06" size=".11 .06" rgba=".2 .24 .28 1"/>
<geom type="capsule" fromto="0 0 .13 {0.62*np.cos(TARGET)} {0.62*np.sin(TARGET)} .13" size=".012" rgba=".1 .6 .3 .55" contype="0" conaffinity="0"/>
<geom type="sphere" pos="{0.62*np.cos(TARGET)} {0.62*np.sin(TARGET)} .13" size=".045" rgba=".1 .6 .3 .35" contype="0" conaffinity="0"/>
<body pos="0 0 .16"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
<joint name="j" type="hinge" axis="0 0 1" damping="{A}" stiffness="{B}" springref="0" armature="0"/>
<geom type="cylinder" size=".07 .04" rgba=".25 .28 .32 1" contype="0" conaffinity="0" mass="0"/>
<geom type="capsule" fromto="0 0 0 .62 0 0" size=".04" rgba="{rgba}" contype="0" conaffinity="0" mass="0"/>
<geom type="sphere" pos=".62 0 0" size=".06" rgba=".2 .24 .28 1" contype="0" conaffinity="0" mass="0"/></body></worldbody>
<actuator><motor joint="j" gear="{K}" ctrllimited="false"/></actuator></mujoco>"""
class Lane:
    def __init__(s, key, name, trials, think_s):
        s.m = mujoco.MjModel.from_xml_string(xml(COL[key])); s.d = mujoco.MjData(s.m); s.r = mujoco.Renderer(s.m, H, W)
        s.cam = mujoco.MjvCamera(); s.cam.lookat[:] = [0.15, 0.2, 0.1]; s.cam.distance = 1.75; s.cam.elevation = -58; s.cam.azimuth = 90
        s.key, s.name, s.trials, s.think = key, name, trials, think_s
        s.i, s.t, s.state, s.done, s.fail_note = 0, 0.0, "think", False, ""
    def reset(s):
        mujoco.mj_resetData(s.m, s.d); s.integ = 0.0; s.df = 0.0; s.eprev = TARGET; s.t = 0.0
    def step(s, dt_wall):  # advance by dt_wall seconds of wall-clock video time
        sim_dt = dt_wall * SPEED
        if s.state == "think":
            s.t += sim_dt
            if s.t >= s.think: s.state = "test"; s.reset()
            return
        if s.state != "test": return
        Kp, Ki, Kd = s.trials[s.i][:3]
        for _ in range(int(round(sim_dt / DT))):
            e = TARGET - s.d.qpos[0]; s.integ += e * DT
            s.df += DT / 0.01 * ((e - s.eprev) / DT - s.df) if Kd else 0.0  # Kd*s/(0.01s+1) filter state
            s.eprev = e; s.d.ctrl[0] = Kp * e + Ki * s.integ + Kd * s.df
            mujoco.mj_step(s.m, s.d); s.t += DT
            if not np.isfinite(s.d.qpos[0]) or abs(s.d.qpos[0]) > 60: s.t = SIMT; break
        if s.t >= SIMT:
            if s.trials[s.i][3]: s.state = "pass"; s.done = True; return
            s.i += 1
            if s.i >= len(s.trials): s.state = "gaveup"; s.done = True; return
            s.reset()
    def frame(s, ff=False):
        q = s.d.qpos[0] if np.isfinite(s.d.qpos[0]) else 0.0; s.d.qpos[0] = q; mujoco.mj_forward(s.m, s.d)
        s.r.update_scene(s.d, s.cam); img = Image.fromarray(s.r.render()); g = ImageDraw.Draw(img)
        n = s.i + 1 if s.state in ("test", "pass") else (len(s.trials) if s.state == "gaveup" else 0)
        g.text((16, 12), s.name, fill=(24, 34, 45), font=F1); g.text((16, 44), f"{n}", fill=tuple(int(255*float(c)) for c in COL[s.key].split()[:3]), font=F2)
        g.text((16 + g.textlength(f"{n}", font=F2) + 10, 74), "step tests", fill=(80, 90, 100), font=F3)
        msg = {"think": f"Thinking… {s.t:4.1f} s", "test": ("Fast-forwarding…" if ff else "Testing…"), "pass": f"Meets spec on try {n}", "gaveup": "Gave up"}[s.state]
        g.text((16, H - 34), msg, fill=(35, 130, 79) if s.state == "pass" else (60, 70, 80), font=F3)
        g.text((W - 150, H - 30), "playback 4× speed", fill=(110, 120, 130), font=F4)
        return np.array(img)
try:
    F1 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24); F2 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 54); F3 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20); F4 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
except Exception: F1 = F2 = F3 = F4 = ImageFont.load_default()
first_c = row["cold_first"] or len(row["cold_log"]); first_o = row["ours_first"]
lanes = [Lane("cold", "Optimizer only", row["cold_log"][:first_c], 0.0),
         Lane("gpt", "GPT-6.1 Terra", [gpt_g + [bool(gpt["success"])]], gpt.get("latency_s", 20)),
         Lane("ours", "Joint Tuner (ours)", row["ours_log"][:first_o], 1.0)]
# sanity: MuJoCo verdict vs judge for each final trial
for L in lanes:
    g = L.trials[-1][:3]; print(L.name, "final gains", np.round(g, 4), "judge:", evaluate(T, *g)["success"])
wr = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
def frames(n, ff_lane=None):
    for _ in range(n):
        for L in lanes:
            if L is ff_lane: continue
            L.step(1 / FPS)
        wr.append_data(np.concatenate([L.frame(ff=(L is ff_lane)) for L in lanes], axis=1))
fr = 0
while not (lanes[1].done and lanes[2].done) and fr < FPS * 40: frames(1); fr += 1
frames(FPS)  # hold
c = lanes[0]
if not c.done:  # fast-forward the optimizer: its remaining trials, shown at high speed
    rem = len(c.trials) - c.i; per = max(1, int(np.ceil(rem / (FPS * 8))))
    while not c.done:
        for _ in range(per):
            c.t = SIMT - 1e-9 if c.state == "test" else c.t; c.step(1e-9) if False else None
            if c.trials[c.i][3]: c.reset(); break
            c.i += 1
            if c.i >= len(c.trials): c.state = "gaveup"; c.done = True; break
        if not c.done and c.trials[c.i][3]:
            c.reset()
            while not c.done: c.step(1 / FPS); wr.append_data(np.concatenate([L.frame() for L in lanes], axis=1))
        else:
            c.reset(); c.step(0.6 / SPEED * SIMT / 3)  # show a slice of this trial's motion
            wr.append_data(np.concatenate([L.frame(ff=(L is c)) for L in lanes], axis=1))
frames(FPS * 2)
wr.close(); print("wrote", OUT)
