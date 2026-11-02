"""Render where a ControlEval plant lives on real hardware. Four MuJoCo scenes; in each, ONE highlighted unit (orange)
is simulated with the task's exact dynamics: inertia 1, joint damping a, joint stiffness c (negative = unstable),
motor gear K; for the drone, a first-order rotor lag via an actuator filter. The rest of the body is fixed for clarity.
The unit is driven by a PID with Joint Tuner's gains (the ones the judge passed). Output: media/units.mp4 + media/units.gif"""
import sys, json, numpy as np, mujoco, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "control")
from judge import load_tasks, evaluate
W, H, FPS, DT = 640, 480, 30, 0.0005
ORANGE, GREY, DARK, STEEL = "1 0.5 0.15 1", ".72 .75 .78 1", ".22 .25 .29 1", ".55 .58 .62 1"
fb = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"; fr = "/System/Library/Fonts/Supplemental/Arial.ttf"
F1, F2, F3 = ImageFont.truetype(fb, 24), ImageFont.truetype(fr, 17), ImageFont.truetype(fb, 17)
TASKS = {t["uid"]: t for t in load_tasks()}
TR = {r["uid"]: r for r in json.load(open("runs/trace_v3.json"))["rows"]}
def ours_gains(uid): r = TR[uid]; return r["ours_log"][r["ours_first"] - 1][:3]
ENV = f"""<option timestep="{DT}" gravity="0 0 0" integrator="RK4"/>
<visual><global offwidth="{W}" offheight="{H}"/><quality shadowsize="4096"/><headlight ambient=".35 .35 .35"/></visual>
<asset><texture name="g" type="2d" builtin="checker" rgb1=".88 .9 .92" rgb2=".82 .84 .87" width="256" height="256"/>
<material name="g" texture="g" texrepeat="10 10" reflectance=".05"/>
<texture type="skybox" builtin="gradient" rgb1=".95 .96 .98" rgb2=".8 .84 .88" width="64" height="64"/></asset>"""
def joint(axis, a, c): return f'<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/><joint name="j" type="hinge" axis="{axis}" damping="{a}" stiffness="{c}"/>'
def ring(axis_euler, r=.09): return f'<geom type="cylinder" size="{r} .012" euler="{axis_euler}" rgba="1 .5 .15 .45" contype="0" conaffinity="0" mass="0"/>'
NG = 'contype="0" conaffinity="0" mass="0"'
SCENES = []
# 1. actuator: motor + gearbox fixed, output flange + pointer is the unit (axis x)
uid = "second_order_stable_fast/6"
SCENES.append(dict(name="Actuator: motor + gearbox output", unit="output shaft position loop", uid=uid, target=1.0, cam=(0.0, 0.0, 0.25, 1.3, -22, 140),
 body=lambda a, c, K: f"""<worldbody><light pos="1 -1 3" dir="-.3 .3 -1"/><geom type="plane" size="3 3 .1" material="g"/>
<geom type="box" pos="0 0 .12" size=".2 .14 .02" rgba="{DARK}"/>
<geom type="cylinder" pos="-.12 0 .25" euler="0 90 0" size=".1 .14" rgba="{STEEL}"/>
<geom type="cylinder" pos="-.12 0 .25" euler="0 90 0" size=".104 .02" rgba="{DARK}"/>
<geom type="box" pos=".07 0 .25" size=".05 .1 .1" rgba="{GREY}"/>
<body pos=".14 0 .25">{joint("1 0 0", a, c)}{ring("0 90 0", .12)}
<geom type="cylinder" euler="0 90 0" size=".09 .015" rgba="{ORANGE}" {NG}/>
<geom type="box" pos=".03 0 .1" size=".012 .02 .1" rgba="{ORANGE}" {NG}/><geom type="sphere" pos=".03 0 .2" size=".025" rgba="{DARK}" {NG}/></body></worldbody>
<actuator><motor joint="j" gear="{K}" ctrllimited="false"/></actuator>"""))
# 2. robot arm: base + upper arm fixed, elbow + forearm + gripper is the unit (axis y)
uid = "second_order_stable_moderate/9"
SCENES.append(dict(name="Robot arm: elbow joint", unit="elbow position loop", uid=uid, target=0.9, cam=(0.25, 0, 0.55, 2.1, -15, 120),
 body=lambda a, c, K: f"""<worldbody><light pos="1 -1 3" dir="-.3 .3 -1"/><geom type="plane" size="3 3 .1" material="g"/>
<geom type="cylinder" pos="0 0 .05" size=".16 .05" rgba="{DARK}"/><geom type="cylinder" pos="0 0 .16" size=".1 .07" rgba="{GREY}"/>
<geom type="sphere" pos="0 0 .25" size=".085" rgba="{GREY}"/>
<geom type="capsule" fromto="0 0 .25 .22 0 .62" size=".055" rgba="{GREY}"/>
<body pos=".22 0 .62">{joint("0 1 0", a, c)}{ring("90 0 0", .1)}
<geom type="sphere" size=".07" rgba="{ORANGE}" {NG}/>
<geom type="capsule" fromto="0 0 0 .42 0 0" size=".045" rgba="{ORANGE}" {NG}/>
<geom type="box" pos=".46 0 0" size=".025 .05 .03" rgba="{DARK}" {NG}/>
<geom type="box" pos=".52 .035 0" size=".05 .01 .02" rgba="{DARK}" {NG}/><geom type="box" pos=".52 -.035 0" size=".05 .01 .02" rgba="{DARK}" {NG}/></body></worldbody>
<actuator><motor joint="j" gear="{K}" ctrllimited="false"/></actuator>"""))
# 3. humanoid: feet fixed, ankle pitch is the unit; everything above the ankle is the inverted pendulum
uid = "second_order_unstable/7"
SCENES.append(dict(name="Humanoid: ankle balance", unit="ankle pitch (inverted pendulum)", uid=uid, target=0.12, cam=(0, 0, .82, 3.4, -4, 125),
 body=lambda a, c, K: f"""<worldbody><light pos="1 -1 4" dir="-.3 .3 -1"/><geom type="plane" size="3 3 .1" material="g"/>
<geom type="box" pos=".05 .1 .025" size=".12 .05 .025" rgba="{DARK}"/><geom type="box" pos=".05 -.1 .025" size=".12 .05 .025" rgba="{DARK}"/>
<body pos="0 0 .08">{joint("0 1 0", a, c)}{ring("90 0 0", .07)}
<geom type="cylinder" fromto="0 .17 0 0 -.17 0" size=".035" rgba="{ORANGE}" {NG}/>
<geom type="capsule" fromto="0 .1 0 0 .1 .8" size=".055" rgba="{GREY}" {NG}/><geom type="capsule" fromto="0 -.1 0 0 -.1 .8" size=".055" rgba="{GREY}" {NG}/>
<geom type="box" pos="0 0 1.1" size=".1 .17 .3" rgba="{GREY}" {NG}/>
<geom type="capsule" fromto="0 .23 1.35 0 .26 .85" size=".04" rgba="{GREY}" {NG}/><geom type="capsule" fromto="0 -.23 1.35 0 -.26 .85" size=".04" rgba="{GREY}" {NG}/>
<geom type="sphere" pos="0 0 1.55" size=".11" rgba="{GREY}" {NG}/><geom type="box" pos=".09 0 1.57" size=".03 .07 .025" rgba="{DARK}" {NG}/></body></worldbody>
<actuator><motor joint="j" gear="{K}" ctrllimited="false"/></actuator>"""))
# 4. drone on a single-axis roll test stand; rotor lag as an actuator filter (plant K/(s^2(s+a)))
dd = json.load(open("runs/drone_demo_gains.json")) if len(sys.argv) < 2 else {}
if dd.get("task"):
    t = dd["task"]
    SCENES.append(dict(name="Drone: roll axis on a test stand", unit="roll attitude loop (with rotor lag)", task=t, gains=dd["tries"][-1][:3], tries=len(dd["tries"]), target=0.35, cam=(0, 0, .6, 2.0, -18, 60), drone=True,
     body=lambda a, c, K: f"""<worldbody><light pos="1 -1 4" dir="-.3 .3 -1"/><geom type="plane" size="3 3 .1" material="g"/>
<geom type="cylinder" fromto="0 -.45 0 0 -.45 .6" size=".03" rgba="{DARK}"/><geom type="cylinder" fromto="0 .45 0 0 .45 .6" size=".03" rgba="{DARK}"/>
<geom type="box" pos="0 0 .01" size=".06 .5 .01" rgba="{DARK}"/>
<body pos="0 0 .6">{joint("1 0 0", 0, 0)}{ring("0 90 0", .08)}
<geom type="cylinder" fromto="0 -.45 0 0 .45 0" size=".012" rgba="{STEEL}" {NG}/>
<geom type="box" size=".09 .09 .035" rgba="{ORANGE}" {NG}/>
<geom type="capsule" fromto="-.24 -.24 0 .24 .24 0" size=".018" rgba="{ORANGE}" {NG}/><geom type="capsule" fromto="-.24 .24 0 .24 -.24 0" size=".018" rgba="{ORANGE}" {NG}/>
<geom type="cylinder" pos=".24 .24 .03" size=".12 .004" rgba=".2 .2 .2 .35" {NG}/><geom type="cylinder" pos="-.24 -.24 .03" size=".12 .004" rgba=".2 .2 .2 .35" {NG}/>
<geom type="cylinder" pos=".24 -.24 .03" size=".12 .004" rgba=".2 .2 .2 .35" {NG}/><geom type="cylinder" pos="-.24 .24 .03" size=".12 .004" rgba=".2 .2 .2 .35" {NG}/></body></worldbody>
<actuator><general joint="j" gear="{K}" dyntype="filter" dynprm="{1.0 / a}" gaintype="fixed" biastype="none" ctrllimited="false"/></actuator>"""))

def run(S):
    t = S.get("task") or TASKS[S["uid"]]; g = S.get("gains") or ours_gains(S["uid"])
    num, den = np.array(t["num"], float), np.array(t["den"], float); num, den = num / den[0], den / den[0]
    if S.get("drone"): a = den[1]; m = mujoco.MjModel.from_xml_string(f"<mujoco>{ENV}{S['body'](a, 0, num[0] / a)}</mujoco>"); a_, c_ = a, 0
    else: a_, c_ = den[1], den[2]; m = mujoco.MjModel.from_xml_string(f"<mujoco>{ENV}{S['body'](a_, c_, num[0])}</mujoco>")
    d = mujoco.MjData(m); r = mujoco.Renderer(m, H, W); cam = mujoco.MjvCamera()
    cam.lookat[:] = S["cam"][:3]; cam.distance, cam.elevation, cam.azimuth = S["cam"][3:]
    ok = evaluate(t, *g)["success"]; Kp, Ki, Kd = g; R = S["target"]; simt = 2.5 * t["settling_time_max"]
    integ = df = 0.0; eprev = R; ys, frames = [], []; SPD = max(1.0, round(simt / 6.0)); steps = int(round(SPD / FPS / DT)); tt = 0.0
    plant = (f"G(s) = {num[0]:.3g} / (s² + {a_:.3g}s {'+' if c_ >= 0 else '−'} {abs(c_):.3g})" if not S.get("drone")
             else f"G(s) = {num[0]:.4g} / (s²(s + {a_:.3g}))")
    def draw():
        img = Image.fromarray(r.render()); gd = ImageDraw.Draw(img, "RGBA")
        gd.rectangle([0, 0, W, 64], fill=(255, 255, 255, 215)); gd.text((16, 8), S["name"], fill=(24, 34, 45), font=F1)
        gd.text((16, 38), f"Highlighted unit: {S['unit']}", fill=(200, 95, 20), font=F3)
        gd.rectangle([0, H - 78, W, H], fill=(255, 255, 255, 215))
        gd.text((16, H - 70), plant + ("   unstable" if c_ < 0 else ""), fill=(24, 34, 45), font=F2)
        gd.text((16, H - 46), f"Joint Tuner gains  Kp {Kp:.3g}   Ki {Ki:.3g}   Kd {Kd:.3g}", fill=(24, 34, 45), font=F2)
        gd.text((16, H - 24), f"{'Meets spec (judge + MuJoCo)' if ok else 'Fails spec'}   settle ≤ {t['settling_time_max']:.2g} s, overshoot ≤ {t['overshoot_max']:.0f}%", fill=(35, 130, 79) if ok else (180, 50, 40), font=F2)
        x0, y0, pw, ph = W - 210, 76, 196, 110  # step-response inset
        gd.rectangle([x0, y0, x0 + pw, y0 + ph], fill=(255, 255, 255, 225), outline=(200, 205, 210))
        Y = lambda v: y0 + ph - 12 - (ph - 24) * (np.clip(v, -.2, 1.6) + .2) / 1.8
        gd.line([x0 + 6, Y(1), x0 + pw - 6, Y(1)], fill=(35, 130, 79), width=1)
        if len(ys) > 1: gd.line([(x0 + 6 + (pw - 12) * i / (simt * FPS / SPD), Y(v)) for i, v in enumerate(ys)], fill=(230, 110, 30), width=2)
        gd.text((x0 + 8, y0 + 4), "step response", fill=(90, 100, 110), font=F2)
        gd.text((W - 170, 40), f"playback {SPD:g}× speed" if SPD > 1 else "real time", fill=(110, 120, 130), font=F2)
        return np.array(img)
    while tt < simt:
        for _ in range(steps):
            e = R - d.qpos[0]; integ += e * DT; df += DT / 0.01 * ((e - eprev) / DT - df); eprev = e
            d.ctrl[0] = Kp * e + Ki * integ + Kd * df; mujoco.mj_step(m, d); tt += DT
        ys.append(d.qpos[0] / R); r.update_scene(d, cam); frames.append(draw())
    for _ in range(FPS): frames.append(frames[-1])
    print(S["name"], "judge pass", ok, "final y", round(ys[-1], 4), "max", round(max(ys), 3), "frames", len(frames))
    return frames
allf = [run(S) for S in SCENES]
SLUG = ["actuator", "robot_arm_elbow", "humanoid_ankle", "drone_roll"]
for slug, fs in zip(SLUG, allf):
    w1 = imageio.get_writer(f"media/{slug}.mp4", fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    for f in fs: w1.append_data(f)
    w1.close(); imageio.mimsave(f"media/{slug}.gif", fs[::3], duration=1000 * 3 / FPS, loop=0)
w = imageio.get_writer("media/units.mp4", fps=FPS, codec="libx264", quality=8, macro_block_size=8)
for fs in allf:
    for f in fs: w.append_data(f)
w.close()
n = max(len(f) for f in allf); grid = []
for i in range(0, n, 3):
    tiles = [fs[min(i, len(fs) - 1)][::2, ::2][::1] for fs in allf] + [np.full((H // 2, W // 2, 3), 240, np.uint8)] * (4 - len(allf))
    grid.append(np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:4], 1)], 0))
imageio.mimsave("media/units_grid.gif", [g[::2, ::2] if False else g for g in grid], duration=1000 * 3 / FPS, loop=0)
print("wrote media/units.mp4 media/units_grid.gif")
