import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import time


# Setting matrix_weight's variables, works good with game field trajectories
Q_x = 5000
Q_y = 5000
Q_theta = 5000
R1 = 1
R2 = 1
R3 = 1
R4 = 1
dt = 0.1
## Robot parameters
wheel_radius = 1
Lx = 0.3
Ly = 0.3
WB = 0.25
index = -3

# Model and MPC params
a1 = math.pi/4
a2 = 3*math.pi/4
a3 = 5*math.pi/4
a4 = 7*math.pi/4
dt = 0.1

# Define the boundary of the problem
x_max = 5.5
x_min = 1.0
y_max = 6
y_min = 1.0
theta_max = np.pi
theta_min = -theta_max
v_max = 1.6666
v_min = -v_max
step_horizon = 0.1 # time between steps in seconds
N = 10 # the prediction horizontal
sim_time = 23.7333 # simulation time

# Condition for simulation time
show_animation = True

# Function to plot game field Robocon 2023

def game_field_plot():
    ## Game field exterior size in [meter]
    gf_coord_x = 0
    gf_coord_y = 0
    gf_width = 12.0
    gf_height = 12.0
    # Rectangle for red side
    rs_coord_x = 0
    rs_corrd_y= 0
    rs_width = 6
    rs_height = 12
    # Rectangle for blue side
    bs_coord_x = 6
    bs_coord_y = 0
    bs_width = 6
    bs_height = 12
    # Rectangle for water area
    wa_coord_x = 1.95
    wa_coord_y = 1.95
    wa_width = 8
    wa_height = 8
    # Rectangle for inside red side
    irs_coord_x = 2.55
    irs_coord_y = 2.55
    irs_width = 3.4
    irs_height = 6.8
    # Rectangle for inside blue side
    ibs_coord_x = 6
    ibs_coord_y = 2.55
    ibs_width = 3.4
    ibs_height = 6.8
    # Rectangle for middle area
    ma_coord_x = 4.45
    ma_coord_y = 4.45
    ma_width = 3
    ma_height = 3
    # Rectagle small for red side
    srs_coord_x = 0
    srs_coord_y = 5.25
    srs_width = 1
    srs_height = 1.5
    # Rectangle small for blue side
    sbs_coord_x = 11
    sbs_coord_y = 5.25
    sbs_width = 1
    sbs_height = 1.5
    # 4 small rectangle white
    # 1
    sr1_coord_x = 0
    sr1_coord_y = 0
    sr1_width = 0.5
    sr1_height = 0.5
    # 2
    sr2_coord_x = 11.5
    sr2_coord_y = 0
    sr2_width = 0.5
    sr2_height = 0.5
    # 3
    sr3_coord_x = 11.5
    sr3_coord_y = 11.5
    sr3_width = 0.5
    sr3_height = 0.5
    # 4
    sr4_coord_x = 0
    sr4_coord_y = 11.5
    sr4_width = 0.5
    sr4_height = 0.5
    # 2 small rectangle yellow
    # 1
    ses1_coord_x = 5.5
    ses1_coord_y = 0
    ses1_width = 0.5
    ses1_height = 0.5
    #2
    ses2_coord_x = 6
    ses2_coord_y = 11.5
    ses2_width = 0.5
    ses2_height = 0.5
    # 2 small rectangle moat area
    #1
    ms1_coord_x = 5.025
    ms1_coord_y = 1.95
    ms1_width = 0.975
    ms1_height = 0.6
    #2
    ms2_coord_x = 6
    ms2_coord_y = 9.35
    ms2_width = 0.975
    ms2_height = 0.6
    # 2 small rectangle in middle area
    #1
    ins1_coord_x = 3.95
    ins1_coord_y = 5.45
    ins1_width = 0.5
    ins1_height = 1.0
    # 2
    ins2_coord_x = 7.45
    ins2_coord_y = 5.45
    ins2_width = 0.5
    ins2_height = 1.0

    # patches game field
    p_game_field = patches.Rectangle((gf_coord_x, gf_coord_y), gf_width, gf_height, facecolor="black")
    # patches rectangle red side
    p_rect_rs = patches.Rectangle((rs_coord_x, rs_corrd_y), rs_width, rs_height, facecolor="red")
    # patches rectangle blue side
    p_rect_bs = patches.Rectangle((bs_coord_x, bs_coord_y), bs_width, bs_height, facecolor="blue")
    # patches rectangle water area
    p_rect_wa = patches.Rectangle((wa_coord_x, wa_coord_y), wa_width, wa_height, facecolor="cyan")
    # patches rectangle inside red area
    p_rect_irs = patches.Rectangle((irs_coord_x, irs_coord_y), irs_width, irs_height, facecolor="red")
    # patches rectangle inside blue area
    p_rect_ibs = patches.Rectangle((ibs_coord_x, ibs_coord_y), ibs_width, ibs_height, facecolor="blue")
    # patches rectangle middle area
    p_rect_ma = patches.Rectangle((ma_coord_x, ma_coord_y), ma_width, ma_height, facecolor="orange")
    # patches rectangle small red side
    p_rect_srs = patches.Rectangle((srs_coord_x, srs_coord_y), srs_width, srs_height, facecolor="red")
    # pacthes rectangle small blue side
    p_rect_sbs = patches.Rectangle((sbs_coord_x, sbs_coord_y), sbs_width, sbs_height, facecolor="blue")
    # patches rectangle small white
    #1
    p_rect_sr1 = patches.Rectangle((sr1_coord_x, sr1_coord_y), sr1_width, sr1_height, facecolor="black")
    #2
    p_rect_sr2 = patches.Rectangle((sr2_coord_x, sr2_coord_y), sr2_width, sr2_height, facecolor="black")
    #3
    p_rect_sr3 = patches.Rectangle((sr3_coord_x, sr3_coord_y), sr3_width, sr3_height, facecolor="black")
    #4
    p_rect_sr4 = patches.Rectangle((sr4_coord_x, sr4_coord_y), sr4_width, sr4_height, facecolor="black")
    # patches 2 small rectangle yellow
    #1
    p_rect_ses1 = patches.Rectangle((ses1_coord_x, ses1_coord_y), ses1_width, ses1_height, facecolor="yellow")
    #2
    p_rect_ses2 = patches.Rectangle((ses2_coord_x, ses2_coord_y), ses2_width, ses2_height, facecolor="yellow")
    # patches 2 small rectangle moat area
    #1
    p_rect_ms1 = patches.Rectangle((ms1_coord_x, ms1_coord_y), ms1_width, ms1_height, facecolor="gray")
    #2
    p_rect_ms2 = patches.Rectangle((ms2_coord_x, ms2_coord_y), ms2_width, ms2_height, facecolor="gray")
    # patches 2 small rectangle in middle area
    #1
    p_rect_ins1 = patches.Rectangle((ins1_coord_x, ins1_coord_y), ins1_width, ins1_height, facecolor="white")
    #2
    p_rect_ins2 = patches.Rectangle((ins2_coord_x, ins2_coord_y), ins2_width, ins2_height, facecolor="white")
    plt.gca().add_patch(p_game_field)
    plt.gca().add_patch(p_rect_rs)
    plt.gca().add_patch(p_rect_bs)
    plt.gca().add_patch(p_rect_wa)
    plt.gca().add_patch(p_rect_irs)
    plt.gca().add_patch(p_rect_ibs)
    plt.gca().add_patch(p_rect_ma)
    plt.gca().add_patch(p_rect_srs)
    plt.gca().add_patch(p_rect_sbs)
    plt.gca().add_patch(p_rect_sr1)
    plt.gca().add_patch(p_rect_sr2)
    plt.gca().add_patch(p_rect_sr3)
    plt.gca().add_patch(p_rect_sr4)
    plt.gca().add_patch(p_rect_ses1)
    plt.gca().add_patch(p_rect_ses2)
    plt.gca().add_patch(p_rect_ms1)
    plt.gca().add_patch(p_rect_ms2)
    plt.gca().add_patch(p_rect_ins1)
    plt.gca().add_patch(p_rect_ins2)

## Function to plot robot
class OMNI_ROBOT:

    def __init__(self, r_h=0.35, r_w=0.35, w_h=0.025, w_w=0.1, rot_angle=45.0, w_pos=0.35):

        # Default parameter for omni robot 4 wheeled
        ## Angle default
        self.a1 = math.pi/4
        self.a2 = 3*math.pi/4
        self.a3 = 5*math.pi/4
        self.a4 = 7*math.pi/4
        ## robot shape
        self.robot_h = r_h
        self.robot_w = r_w
        ## wheel shaspe
        self.wheel_h = w_h
        self.wheel_w = w_w
        ## wheel position
        self.w_pos = w_pos
        ## Rotation angle
        self.rotate_angle = rot_angle

    def rotation_angle(self, theta):
        matrix_transform = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]]).T
        return matrix_transform


    def robot_shape(self):
        matrix_shape = np.array([
            [-self.robot_w, self.robot_w, self.robot_w, -self.robot_w, -self.robot_w],
            [self.robot_h, self.robot_h, -self.robot_h, -self.robot_h, self.robot_h]
        ])
        return matrix_shape

    def robot_wheel(self):
        matrix_wheel = np.array([
            [-self.wheel_w, self.wheel_w, self.wheel_w, -self.wheel_w, -self.wheel_w],
            [self.wheel_h, self.wheel_h, -self.wheel_h, -self.wheel_h, self.wheel_h]
        ])
        return matrix_wheel

    def generate_each_wheel_and_draw(self, x, y, yaw):
        # Copy Matrix as robot_shape
        pos_wheel = self.robot_wheel()
        pos_wheel1 = pos_wheel.copy()
        pos_wheel2 = pos_wheel.copy()
        pos_wheel3 = pos_wheel.copy()
        pos_wheel4 = pos_wheel.copy()
        # Transpose wheel to each small length
        # pos_wheel1 = np.dot(pos_wheel1.T, self.rotation_angle(-self.rotate_angle)).T
        # pos_wheel2 = np.dot(pos_wheel2.T, self.rotation_angle(self.rotate_angle)).T
        # pos_wheel3 = np.dot(pos_wheel3.T, self.rotation_angle(-self.rotate_angle)).T
        # pos_wheel4 = np.dot(pos_wheel4.T, self.rotation_angle(self.rotate_angle)).T
        # Push each wheel to where it belong
        pos_wheel1[0, :] += self.w_pos
        pos_wheel1[1, :] -= self.w_pos
        pos_wheel2[0, :] += self.w_pos
        pos_wheel2[1, :] += self.w_pos
        pos_wheel3[0, :] -= self.w_pos
        pos_wheel3[1, :] += self.w_pos
        pos_wheel4[0, :] -= self.w_pos
        pos_wheel4[1, :] -= self.w_pos
        # Matrix Transforms each wheel 1, 2, 3, 4
        pos_wheel1 = np.dot(pos_wheel1.T, self.rotation_angle(yaw)).T
        pos_wheel2 = np.dot(pos_wheel2.T, self.rotation_angle(yaw)).T
        pos_wheel3 = np.dot(pos_wheel3.T, self.rotation_angle(yaw)).T
        pos_wheel4 = np.dot(pos_wheel4.T, self.rotation_angle(yaw)).T

        # Matrix Transforms robot shape
        robot_shape = self.robot_shape()
        robot_shaped = np.dot(robot_shape.T, self.rotation_angle(yaw)).T

        plt.plot(robot_shaped[0, :]+x, robot_shaped[1, :]+y, color="blue")
        plt.plot(pos_wheel1[0, :]+x, pos_wheel1[1, :]+y, color="black")
        plt.plot(pos_wheel2[0, :]+x, pos_wheel2[1, :]+y, color="black")
        plt.plot(pos_wheel3[0, :]+x, pos_wheel3[1, :]+y, color="black")
        plt.plot(pos_wheel4[0, :]+x, pos_wheel4[1, :]+y, color="black")

# Function to animation

def animation_robot(hist_x, hist_y, hist_th, vx, vy, vth, index_x, index_y, time_counts):
    omni_robot = OMNI_ROBOT()
    w = np.linspace(0, 25, 100)
    #ax = 3 + 3 * np.sin(2*np.pi*w/25)
    #ay = 2*np.sin(4*np.pi*w/25)
    ax = [1, 1, 5.5, 5.5, 3.5, 3.5]
    ay = [6, 1, 1, 3.6, 3.6, 6]

    plt.figure(figsize=(12, 8))
    for t in range(len(time_counts)):
        x = hist_x[t]
        y = hist_y[t]
        theta = hist_th[t]
        vth = vth.flatten()
        plt.clf()
        plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
        #plt.scatter(next_trajectories[0, :], next_trajectories[1, :], color="red")
        plt.plot(ax, ay, color="black")
        #plt.plot([0, 3], [1, 1], color="green")
        game_field_plot()
        omni_robot.generate_each_wheel_and_draw(x, y, theta)
        #plot_car(x, y, theta)
        plot_arrow(x, y, theta)
        #plt.plot(next_trajectories[0, -1:], next_trajectories[1, -1:])
        plt.plot(index_x[t], index_y[t], marker="*", color="black")
        plt.axis("equal")
        #plt.grid(True)
        plt.title("Linear velocity :" + str(round(math.sqrt(vx[t]**2+vy[t]**2), 2)) + " m/s")
        plt.pause(0.01)

# Function to plot robot
def plot_car(x, y, yaw, steer=0.0, truck_color="-k"):  # pragma: no cover

    # Vehicle parameters
    LENGTH = 0.8  # [m]
    WIDTH = 0.4  # [m]
    BACK_TO_WHEEL = 0.1  # [m]
    WHEEL_LEN = 0.7  # [m]
    WHEEL_WIDTH = 0.2  # [m]
    TREAD = 0.5  # [m]

    outline = np.array(
        [[-BACK_TO_WHEEL, (LENGTH - BACK_TO_WHEEL), (LENGTH - BACK_TO_WHEEL),
          -BACK_TO_WHEEL, -BACK_TO_WHEEL],
         [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array(
        [[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH -
          TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truck_color)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truck_color)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truck_color)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truck_color)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truck_color)
    plt.plot(x, y, "*")


# Function to plot Arrow
def plot_arrow(x, y, yaw, length=0.05, width=0.3, fc="b", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

# The forward kinematic model
def forward_kinematic(v1, v2, v3, v4):
    vx = (v1+v2+v3+v4)*(wheel_radius/4)
    vy = (-v1+v2+v3-v4)*(wheel_radius/4)
    vth = (-v1+v2-v3+v4)*(wheel_radius/4)*(1/(Lx+Ly))

    return vx, vy, vth


# The inverse kinematic model
def inverse_kinematic(v_x, v_y, v_theta):

    v1 = (1/wheel_radius)*(v_x-v_y-(Lx+Ly)*v_theta)
    v2 = (1/wheel_radius)*(v_x+v_y+(Lx+Ly)*v_theta)
    v3 = (1/wheel_radius)*(v_x+v_y-(Lx+Ly)*v_theta)
    v4 = (1/wheel_radius)*(v_x-v_y+(Lx+Ly)*v_theta)

    return v1, v2, v3, v4

# Function to calculate velocity using discrete set points
def velocity_from_discrete_points(k, dt, x_ref, y_ref, theta_ref):
    vx = (x_ref[k]-x_ref[k-1])/dt
    vy = (y_ref[k]-y_ref[k-1])/dt
    vth = (theta_ref[k]-theta_ref[k-1])/dt

    return vx, vy, vth

# Shift timestep at each iteration
def shift_timestep(step_horizon, t0, x0, x_f, u, f):
    x0 = x0.reshape((-1,1))
    t = t0 + step_horizon
    f_value = f(x0, u[:, 0])
    st = ca.DM.full(x0 + (step_horizon) * f_value)
    u = np.concatenate((u[:, 1:], u[:, -1:]), axis=1)
    x_f = np.concatenate((x_f[:, 1:], x_f[:, -1:]), axis=1)
    return t, st, x_f, u

# Function to calculate reference trajectory
def reference_state_and_control(t, step_horizon, x0, N, type="None"):
    # initial state
    x = x0.reshape(1, -1).tolist()[0]
    u = []
    # set points for calculating reference control
    x_ref, y_ref, theta_ref = [], [], []
    x_ref_, y_ref_, theta_ref_ = 0, 0, 0
    vx, vy, vth = 0, 0, 0
    # state for N predictions
    for k in range(N):
        t_predict = t + step_horizon * k
        if type == "circle":
            angle = math.pi/10*t_predict
            x_ref_ = 5*math.cos(angle)
            y_ref_ = 5*math.sin(angle)
            theta_ref_ = 0.0
            # Inserting all set points
            x.append(x_ref_)
            x.append(y_ref_)
            x.append(theta_ref_)
            x_ref.append(x_ref_)
            y_ref.append(y_ref_)
            theta_ref.append(theta_ref_)
            vx, vy, vth = velocity_from_discrete_points(k, 0.1, x_ref, y_ref, theta_ref)
            if t_predict >= 18.0:
                vx, vy = 0.0, 0.0
            v_ref1, v_ref2, v_ref3, v_ref4 = inverse_kinematic(vx, vy, vth, theta_ref[k])
            u.append(v_ref1)
            u.append(v_ref2)
            u.append(v_ref3)
            u.append(v_ref4)

        if type == "8shaped":
            x_ref_ = 3 + 3 * np.sin(2*np.pi*t_predict/25)
            y_ref_ = 2*np.sin(4*np.pi*t_predict/25)
            theta_ref_ = np.pi/2
            # Inserting all set points
            x.append(x_ref_)
            x.append(y_ref_)
            x.append(theta_ref_)
            x_ref.append(x_ref_)
            y_ref.append(y_ref_)
            theta_ref.append(theta_ref_)
            vx, vy, vth = velocity_from_discrete_points(k, 0.1, x_ref, y_ref, theta_ref)
            if t_predict >= 18.0:
                vx, vy = 0.0, 0.0
            v_ref1, v_ref2, v_ref3, v_ref4 = inverse_kinematic(vx, vy, vth, theta_ref[k])
            u.append(v_ref1)
            u.append(v_ref2)
            u.append(v_ref3)
            u.append(v_ref4)
        if type == "traj1":
            #if (t_predict <= 10.05):
            #    x_ref_ = 0.975
            #    y_ref_ = 6-0.5*t_predict
            #    theta_ref_ = 0.0
            #if ((t_predict > 10.05) and (t_predict <= 18.1)):
            #    x_ref_ = 0.5*t_predict-5.025+0.975
            #    y_ref_ = 0.975
            #    theta_ref_ = (np.pi/2)/18.1*t_predict
            #if ((t_predict > 18.1) and (t_predict <= 20.9)):
            #    x_ref_ = 5
            #    y_ref_ = 0.5*t_predict - 9.05 + 0.975
            #    theta_ref_ = np.pi/2
            #if ((t_predict > 20.9) and (t_predict <= 24.2)):
            #    x_ref_ = 5-0.5*t_predict+10.45
            #    y_ref_ = 2.375
            #    theta_ref_ = np.pi/2
            #if ((t_predict > 24.2) and (t_predict <= 31.45)):
            #    x_ref_ = 3.35
            #    y_ref_ = 0.5*t_predict-12.1+2.375
            #    theta_ref_ = np.pi/2-((np.pi/2)/31.45*t_predict)
            if (t_predict <= 3.35):
                x_ref_ = 0.975
                y_ref_ = 6-1.5*t_predict
                theta_ref_ = 0.0 #3*np.pi/2
            if ((t_predict > 3.35) and (t_predict <= 6.3666)):
                x_ref_ = 1.5*t_predict - 5.025 +0.975
                y_ref_ = 0.975
                theta_ref_ = 0.0 #4*np.pi/2
            if ((t_predict > 6.3666) and (t_predict <= 7.94933)):
                x_ref_ = 5.5
                y_ref_ = 1.5*t_predict+0.975-9.549
                theta_ref_ = 0.0 #5*np.pi/2
            if ((t_predict > 7.94933) and (t_predict <= 9.3826)):
                x_ref_ = 5.5-1.5*t_predict+11.9239
                y_ref_ = 2.375+0.975
                theta_ref_ = 0.0# 6*np.pi
            if ((t_predict > 9.3826) and (t_predict <= 11.79)):
                x_ref_ = 3.35
                y_ref_ = 1.5*t_predict-14.0739+2.375+0.975
                theta_ref_ = 0.0# 8*np.pi/2
            x.append(x_ref_)
            x.append(y_ref_)
            x.append(theta_ref_)
            x_ref.append(x_ref_)
            y_ref.append(y_ref_)
            theta_ref.append(theta_ref_)
            vx, vy, vth = velocity_from_discrete_points(k, 0.1, x_ref, y_ref, theta_ref)
            if (t_predict >= 11.5):
                vx, vy = 0.0, 0.0
            v_ref1, v_ref2, v_ref3, v_ref4 = inverse_kinematic(vx, vy, vth)
            u.append(v_ref1)
            u.append(v_ref2)
            u.append(v_ref3)
            u.append(v_ref4)

        if type=="traj2":
            if (t_predict <= 3):
                x_ref_ = 1
                y_ref_ = 6-1.6666*t_predict
                theta_ref_ = -np.pi/2
            if ((t_predict > 3) and (t_predict <= 5)):
                x_ref_ = 1
                y_ref_ = 1
                theta_ref_ = -np.pi/2+0.7853*t_predict-3*0.7853
            if ((t_predict > 5) and (t_predict <=8)):
                x_ref_ = 1 + 1.5*t_predict - 1.5*5
                y_ref_ = 1
                theta_ref_ = 0
            if ((t_predict > 8) and (t_predict <= 10)):
                x_ref_ = 5.5
                y_ref_ = 1
                theta_ref_ = 0.7853*t_predict - 0.7853*8
            if ((t_predict > 10) and (t_predict <=13)):
                x_ref_ = 5.5
                y_ref_ = 1 + 0.8666*t_predict - 0.8666*10
                theta_ref_ = np.pi/2
            if ((t_predict > 13) and (t_predict <= 15)):
                x_ref_ = 5.5
                y_ref_ = 3.6
                theta_ref_ = np.pi/2 + 0.7853*t_predict - 0.7853*13
            if ((t_predict > 15) and (t_predict <= 18)):
                x_ref_ = 5.5-0.6666*t_predict+0.6666*15
                y_ref_ = 3.6
                theta_ref_ = np.pi
            if ((t_predict > 18) and (t_predict <= 20)):
                x_ref_ = 3.5
                y_ref_ = 3.6
                theta_ref_ = np.pi-0.7853*t_predict+0.7853*18
            if ((t_predict > 20) and (t_predict <= 23)):
                x_ref_ = 3.5
                y_ref_ = 3.6 + 0.8*t_predict - 0.8*20
                theta_ref_ = np.pi/2
            if ((t_predict > 23) and (t_predict <=25)):
                x_ref_ = 3.5
                y_ref_ = 6
                theta_ref_ = np.pi/2-0.7853*t_predict+0.7853*23
            x.append(x_ref_)
            x.append(y_ref_)
            x.append(theta_ref_)
            x_ref.append(x_ref_)
            y_ref.append(y_ref_)
            theta_ref.append(theta_ref_)
            vx, vy, vth = velocity_from_discrete_points(k, 0.1, x_ref, y_ref, theta_ref)
            if (t_predict >= 24.8):
                vx, vy = 0.0, 0.0
            v_ref1, v_ref2, v_ref3, v_ref4 = inverse_kinematic(vx, vy, vth)
            u.append(v_ref1)
            u.append(v_ref2)
            u.append(v_ref3)
            u.append(v_ref4)
        if type=="traj7":
            ## Phase 1
            if (t_predict <= 3):
                x_ref_ = 1
                y_ref_ = 6-1.6666*t_predict
                theta_ref_ = -np.pi/2
            if ((t_predict > 3) and (t_predict <= 5)):
                x_ref_ = 1
                y_ref_ = 1
                theta_ref_ = -np.pi/2+0.7853*t_predict-3*0.7853
            ## Phase 2
            if ((t_predict > 5) and (t_predict <=8)):
                x_ref_ = 1 + 1.5*t_predict - 1.5*5
                y_ref_ = 1
                theta_ref_ = 0
            if ((t_predict > 8) and (t_predict <= 10)):
                x_ref_ = 5.5
                y_ref_ = 1
                theta_ref_ = 0.7853*t_predict - 0.7853*8
            ## Phase 3
            if ((t_predict > 10) and (t_predict <=11.7333)):
                x_ref_ = 5.5
                y_ref_ = 1 + 1.5*t_predict - 1.5*10
                theta_ref_ = np.pi/2
            if ((t_predict > 11.7333) and (t_predict <= 13.7333)):
                x_ref_ = 5.5
                y_ref_ = 3.6
                theta_ref_ = np.pi/2 + 0.7853*t_predict - 0.7853*11.7333
            ## Phase 4
            if ((t_predict > 13.7333) and (t_predict <= 16.7333)):
                x_ref_ = 5.5-0.6666*t_predict+0.6666*13.7333
                y_ref_ = 3.6
                theta_ref_ = np.pi
            if ((t_predict > 16.7333) and (t_predict <= 18.7333)):
                x_ref_ = 3.5
                y_ref_ = 3.6
                theta_ref_ = np.pi-0.7853*t_predict+0.7853*16.7333
            ## Phase 5
            if ((t_predict > 18.7333) and (t_predict <= 21.7333)):
                x_ref_ = 3.5
                y_ref_ = 3.6 + 0.8*t_predict - 0.8*18.7333
                theta_ref_ = np.pi/2
            if ((t_predict > 21.7333) and (t_predict <=23.7333)):
                x_ref_ = 3.5
                y_ref_ = 6
                theta_ref_ = np.pi/2-0.7853*t_predict+0.7853*21.733
            x.append(x_ref_)
            x.append(y_ref_)
            x.append(theta_ref_)
            x_ref.append(x_ref_)
            y_ref.append(y_ref_)
            theta_ref.append(theta_ref_)
            vx, vy, vth = velocity_from_discrete_points(k, 0.1, x_ref, y_ref, theta_ref)
            if (t_predict >= 23):
                vx, vy = 0.0, 0.0
            v_ref1, v_ref2, v_ref3, v_ref4 = inverse_kinematic(vx, vy, vth)
            u.append(v_ref1)
            u.append(v_ref2)
            u.append(v_ref3)
            u.append(v_ref4)

    # reshaped state and control
    x = np.array(x).reshape(N+1, -1)
    u = np.array(u).reshape(N, -1)

    return x, u


# State symbolic variables
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(x, y, theta)
n_states = states.numel()

# Control symbolic variables
v1 = ca.SX.sym('v1')
v2 = ca.SX.sym('v2')
v3 = ca.SX.sym('v3')
v4 = ca.SX.sym('v4')
controls = ca.vertcat(v1, v2, v3, v4)
n_controls = controls.numel()

# Matrix containing all states
X = ca.SX.sym('X', n_states, N+1)
# Matrix containing all controls
U = ca.SX.sym('U', n_controls, N)
# Matrix containing all states_ref
X_ref = ca.SX.sym('X_ref', n_states, N+1)
# Matrix containing all controls_ref
U_ref = ca.SX.sym('U_ref', n_controls, N)
# State weight matrix
Q = ca.diagcat(Q_x, Q_y, Q_theta)
# Control weight matrix
R = ca.diagcat(R1, R2, R3, R4)
# Rotation matrix
rot_3d_z = ca.vertcat(
    ca.horzcat(ca.cos(theta), -ca.sin(theta), 0),
    ca.horzcat(ca.sin(theta),  ca.cos(theta), 0),
    ca.horzcat(         0,           0, 1)
)
# Matrix transformation
J = (wheel_radius/4) * ca.DM([
    [         1,         1,          1,         1],
    [        -1,         1,          1,        -1],
    [-1/(Lx+Ly), 1/(Lx+Ly), -1/(Lx+Ly), 1/(Lx+Ly)]
])

# RHS
RHS = rot_3d_z @ J @ controls

# nonlinear function for mpc model
f = ca.Function('f', [states, controls], [RHS])
# cost and constraint
cost_fn = 0
g = X[:, 0] - X_ref[:, 0]

# Euler
for k in range(N):
    st_err = X[:, k] - X_ref[:, k+1]
    con_err = U[:, k] - U_ref[:, k]
    cost_fn = cost_fn + st_err.T @ Q @ st_err + con_err.T @ R @ con_err
    st_next = X[:, k+1]
    f_value = f(X[:, k], U[:, k])
    st_next_euler = X[:, k] + (step_horizon*f_value)
    g = ca.vertcat(g, st_next-st_next_euler)

# Making decision variable and optimal parameters
optimal_var = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
optimal_par = ca.vertcat(X_ref.reshape((-1, 1)), U_ref.reshape((-1, 1)))
nlp_prob = {'f': cost_fn, 'x': optimal_var, 'p': optimal_par, 'g': g}
opts = {
        'ipopt.max_iter': 2000,
        'ipopt.print_level': False,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.acceptable_obj_change_tol': 1e-6,
        'print_time': 0}
# Use Interior point optimization
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# Setting all essential boundary
lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

# Lower boundary for decision variable states
lbx[0: n_states*(N+1): n_states] = x_min
lbx[1: n_states*(N+1): n_states] = y_min
lbx[2: n_states*(N+1): n_states] = theta_min
# Upper boundary for decision variable states
ubx[0: n_states*(N+1): n_states] = x_max
ubx[1: n_states*(N+1): n_states] = y_max
ubx[2: n_states*(N+1): n_states] = theta_max

# Lower and upper boundary for variable control
lbx[n_states*(N+1):] = v_min
ubx[n_states*(N+1):] = v_max

args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)),
    'ubg': ca.DM.zeros((n_states*(N+1), 1)),
    'lbx': lbx,
    'ubx': ubx
}

# simulation
t0 = 0
mpciter = 0
init_state = np.array([1, 6.0, -np.pi/2]).reshape(1,-1)
current_state = init_state.copy()
init_control = np.array([0.0, 0.0, 0.0, 0.0]).reshape(1, -1)
state = np.tile(current_state.reshape(-1, 1), N+1).T
control = np.tile(init_control.reshape(-1,1), N).T
next_trajectories = state.copy()
next_controls = control.copy()
opt_x_x = []
opt_x_y = []
opt_x_th = []
index_pre_x = []
index_pre_y = []
t = []
vel_x, vel_y , vel_th = [], [], []
vel_m1, vel_m2, vel_m3, vel_m4 = [], [], [], []
th_cond = [0]

# Generation plot for 8shaped trajectory
w = np.linspace(0, 25, 100)
#ax = 3 + 3 * np.sin(2*np.pi*w/25)
#ay = 2*np.sin(4*np.pi*w/25)
ax = [0.975, 0.975, 5.5, 5.5, 3.35, 3.35]
ay = [6, 0.975, 0.975, 2.375+0.975, 2.375+0.975, 6]
theta_bool = True
t1 = time.time()
while ((mpciter  * step_horizon < sim_time)):
        args['p'] = np.concatenate((
            next_trajectories.reshape((-1, 1)),
            next_controls.reshape((-1, 1)))
        )
        args['x0'] = np.concatenate(
            (state.reshape((-1,1)),
            control.reshape((-1,1)))
        )
        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p = args['p']
        )
        sol_x = ca.reshape(sol['x'][:n_states*(N+1)], n_states, N+1)
        sol_u = ca.reshape(sol['x'][n_states*(N+1):], n_controls, N)
        opt_x_x.append(sol_x[0, -2].full())
        opt_x_y.append(sol_x[1, -2].full())
        opt_x_th.append(sol_x[2, -2].full())
        index_pre_x.append(sol_x[0, -2].full())
        index_pre_y.append(sol_x[1, -2].full())
        t0, current_state, state, control = shift_timestep(step_horizon, t0, current_state, sol_x, sol_u, f)
        next_trajectories, next_controls = reference_state_and_control(t0, step_horizon, current_state, N, type="traj7")
        v_x, v_y, v_th = forward_kinematic(sol_u[0, -2].full(), sol_u[1, -2].full(), sol_u[2, -2].full(), sol_u[3, -2].full())
        vel_x.append(v_x)
        vel_y.append(v_y)
        vel_th.append(v_th)
        vel_m1.append(sol_u[0, -2].full())
        vel_m2.append(sol_u[1, -2].full())
        vel_m3.append(sol_u[2, -2].full())
        vel_m4.append(sol_u[3, -2].full())
        print(f" optimal state :x = {np.round(sol_x.full()[0, 0], 3)}, y = {np.round(sol_x.full()[1, 0], 3)}, theta = {np.round(sol_x.full()[2, 0], 3)}")
        print(f" current state :x = {np.round(current_state[0], 3)}, y = {np.round(current_state[1], 3)}, theta = {np.round(current_state[2], 3)}")
        #print(np.array(opt_x_x).flatten())
        mpciter = mpciter + 1
        t.append(t0)
        #next_trajectories, next_control = reference_trajectory(t0, step_horizon, init_state, N, dt=0.1)
        #print(t0)
t2 = time.time()

t = np.array(t).flatten()
vel_x = np.array(vel_x).flatten()
vel_y = np.array(vel_y).flatten()
vel_th = np.array(vel_th).flatten()
vel_m1 = np.array(vel_m1).flatten()
vel_m2 = np.array(vel_m2).flatten()
vel_m3 = np.array(vel_m3).flatten()
vel_m4 = np.array(vel_m4).flatten()
opt_x_x = np.array(opt_x_x).flatten()
opt_x_y = np.array(opt_x_y).flatten()
opt_x_th = np.array(opt_x_th).flatten()
index_pre_x = np.array(index_pre_x).flatten()
index_pre_y = np.array(index_pre_y).flatten()
print("Total time: {}".format(t2-t1))
if show_animation == True:
    animation_robot(opt_x_x, opt_x_y, opt_x_th, vel_x, vel_y, vel_th, index_pre_x, index_pre_y, t)
    fig, axes = plt.subplots(3, 3, layout="constrained", figsize=(12, 7))
    axes[0, 0].plot(ax, ay, color="red")
    axes[0, 0].set_xlabel('x [m]')
    axes[0, 0].set_ylabel('y [m]')
    axes[0, 0].set_title("Reference trajectory")
    axes[0, 0].grid(True)
    axes[0, 1].plot(opt_x_x, opt_x_y, color="blue")
    axes[0, 1].set_xlabel('x [m]')
    axes[0, 1].set_ylabel('y [m]')
    axes[0, 1].set_title("Exact tracking")
    axes[0, 1].grid(True)
    axes[0, 2].plot(t, vel_x, color="red")
    axes[0, 2].set_xlabel('t (s)')
    axes[0, 2].set_ylabel('m/s')
    axes[0, 2].set_title("Horizontal velocity")
    axes[0, 2].grid(True)
    axes[1, 0].plot(t, vel_y, color="blue")
    axes[1, 0].set_xlabel('t (s)')
    axes[1, 0].set_ylabel('m/s')
    axes[1, 0].set_title("Vertical velocity")
    axes[1, 0].grid(True)
    axes[1, 1].plot(t, vel_th, color="green")
    axes[1, 1].set_xlabel('t (s)')
    axes[1, 1].set_ylabel('m/s')
    axes[1, 1].set_title("Angular velocity")
    axes[1, 1].grid(True)
    axes[1, 2].plot(t, vel_m1, color="green")
    axes[1, 2].set_xlabel('t (s)')
    axes[1, 2].set_ylabel('m/s')
    axes[1, 2].set_title("Velocity of motor 1")
    axes[1, 2].grid(True)
    axes[2, 0].plot(t, vel_m2, color="red")
    axes[2, 0].set_xlabel('t (s)')
    axes[2, 0].set_ylabel('m/s')
    axes[2, 0].set_title("Velocity of motor 2")
    axes[2, 0].grid(True)
    axes[2, 1].plot(t, vel_m3, color="blue")
    axes[2, 1].set_xlabel('t (s)')
    axes[2, 1].set_ylabel('m/s')
    axes[2, 1].set_title("Velocity of motor 3")
    axes[2, 1].grid(True)
    axes[2, 2].plot(t, vel_m4, color="orange")
    axes[2, 2].set_xlabel('t (s)')
    axes[2, 2].set_ylabel('m/s')
    axes[2, 2].set_title("Velocity of motor 4")
    axes[2, 2].grid(True)
    plt.show()
