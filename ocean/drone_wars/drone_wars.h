// Drone Wars - Two-drone aerial combat with 9mm ballistics
// Based on the original drone implementation by Sam Turner and Finlay Sanders, 2025.
// Extended for combat simulation
//
// Observations (34 per drone):
//   own state (14): body-frame velocity, angular velocity, quaternion, motor RPMs
//   opponent (9): relative position, relative velocity, disabled, crashed, bullets remaining
//   combat status (2): own bullets remaining, own disabled
//   incoming bullets (9): relative positions of up to 3 enemy bullets
// Actions (5 continuous per drone): 4 motor commands in [-1, 1], fire when > 0

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NUM_DRONES 2
#define OBS_SIZE 34
#define NUM_ATNS 5
#define ACT_SIZES {1, 1, 1, 1, 1}
#define PUF_STEPS_PER_SEC 50 // 1 / DT

// Visualisation properties
#define WIDTH 1080
#define HEIGHT 720
#define TRAIL_LENGTH 50
#define HORIZON 2048

// Physical constants for the drone (same as original)
#define BASE_MASS 1.0f       // kg
#define BASE_IXX 0.01f       // kgm^2
#define BASE_IYY 0.01f       // kgm^2
#define BASE_IZZ 0.02f       // kgm^2
#define BASE_ARM_LEN 0.1f    // m
#define BASE_K_THRUST 3e-5f  // thrust coefficient
#define BASE_K_ANG_DAMP 0.2f // angular damping coefficient
#define BASE_K_DRAG 1e-6f    // drag (torque) coefficient
#define BASE_B_DRAG 0.1f     // linear drag coefficient
#define BASE_GRAVITY 9.81f   // m/s^2
#define BASE_MAX_RPM 750.0f  // rad/s
#define BASE_MAX_VEL 50.0f   // m/s
#define BASE_MAX_OMEGA 50.0f // rad/s
#define BASE_K_MOT 0.1f      // s (Motor lag constant)
#define BASE_J_MOT 1e-5f     // kgm^2 (Motor rotational inertia)

// Combat-specific constants
#define MAX_BULLETS 3        // Maximum bullets per drone
#define BULLET_MASS 0.0075f  // 9mm bullet mass in kg (7.5 grams)
#define BULLET_VELOCITY 400.0f // 9mm muzzle velocity in m/s
#define BULLET_DRAG 0.0001f  // Bullet drag coefficient
#define HIT_RADIUS 0.5f      // Collision radius for bullet hits
#define CRITICAL_HIT_THRESHOLD 10.0f // Impact threshold to disable drone immediately (m/s)

// Simulation properties
#define GRID_X 50.0f
#define GRID_Y 50.0f
#define GRID_Z 30.0f
#define MARGIN_X (GRID_X - 2)
#define MARGIN_Y (GRID_Y - 2)
#define MARGIN_Z (GRID_Z - 2)
#define DT 0.02f
#define DT_RNG 0.0f

// Camera control constants
#define CAMERA_DISTANCE 100.0f
#define CAMERA_ANGLE_SPEED 2.0f  // degrees per frame
#define CAMERA_MIN_PITCH 10.0f   // minimum pitch angle (degrees)
#define CAMERA_MAX_PITCH 85.0f   // maximum pitch angle (degrees)

// Flat float struct: the trainer sums it across envs and divides by n
struct Log {
    float perf;   // win rate
    float score;  // win rate
    float episode_return;
    float episode_length;
    float hits_scored;
    float hits_received;
    float bullets_fired;
    float accuracy;
    float n;
};

typedef struct {
    float w, x, y, z;
} Quat;

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    Vec3 pos;      // Current position
    Vec3 vel;      // Current velocity
    bool active;   // Whether bullet is in flight
    int owner;     // Which drone fired this (0 or 1)
} Bullet;

typedef struct {
    Vec3 pos;      // global position (x, y, z)
    Vec3 vel;      // linear velocity (u, v, w)
    Quat quat;     // roll/pitch/yaw (phi/theta/psi) as a quaternion
    Vec3 omega;    // angular velocity (p, q, r)
    float rpms[4]; // motor RPMs
} State;

typedef struct {
    Vec3 vel;         // Derivative of position
    Vec3 v_dot;       // Derivative of velocity
    Quat q_dot;       // Derivative of quaternion
    Vec3 w_dot;       // Derivative of angular velocity
    float rpm_dot[4]; // Derivative of motor RPMs
} StateDerivative;

typedef struct {
    float mass;
    float ixx;
    float iyy;
    float izz;
    float arm_len;
    float k_thrust;
    float k_ang_damp;
    float k_drag;
    float b_drag;
    float gravity;
    float max_rpm;
    float max_vel;
    float max_omega;
    float k_mot;
    float j_mot;
} Params;

typedef struct {
    // Core state and parameters
    State state;
    Params params;
    Vec3 prev_pos;

    // Combat state
    Bullet bullets[MAX_BULLETS];
    int bullets_remaining;
    bool disabled;           // Drone is disabled but can still fire
    bool crashed;            // Drone has hit the ground
    float hit_impact;        // Impact velocity when hit
    float ground_height;     // Z position when crashed

    // Logging
    float episode_return;
    int episode_length;
    int hits_scored;
    int hits_received;
    int bullets_fired;
} CombatDrone;

typedef struct Client {
    int width;
    int height;
    bool show_trails;
    bool show_bullets;
    bool paused;
    Camera3D camera;
    float camera_yaw;    // horizontal angle in degrees
    float camera_pitch;  // vertical angle in degrees
} Client;

struct Env {
    Log log;
    Agent agents[NUM_DRONES];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
    int tick;
    CombatDrone drones[NUM_DRONES];
    Client* client;
};
typedef Env DroneWarsEnv;

// Math utility functions
static inline float clampf(float v, float min, float max) {
    if (v < min) return min;
    if (v > max) return max;
    return v;
}

static inline float rndf(float a, float b, unsigned int* rng) {
    return a + ((float)rand_r(rng) / (float)RAND_MAX) * (b - a);
}

static inline Vec3 add3(Vec3 a, Vec3 b) {
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Quat add_quat(Quat a, Quat b) {
    return (Quat){a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 sub3(Vec3 a, Vec3 b) {
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Vec3 scalmul3(Vec3 a, float b) {
    return (Vec3){a.x * b, a.y * b, a.z * b};
}

static inline Quat scalmul_quat(Quat a, float b) {
    return (Quat){a.w * b, a.x * b, a.y * b, a.z * b};
}

static inline float dot3(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float norm3(Vec3 a) {
    return sqrtf(dot3(a, a));
}

static inline void clamp3(Vec3 *vec, float min, float max) {
    vec->x = clampf(vec->x, min, max);
    vec->y = clampf(vec->y, min, max);
    vec->z = clampf(vec->z, min, max);
}

static inline void clamp4(float a[4], float min, float max) {
    a[0] = clampf(a[0], min, max);
    a[1] = clampf(a[1], min, max);
    a[2] = clampf(a[2], min, max);
    a[3] = clampf(a[3], min, max);
}

static inline Quat quat_mul(Quat q1, Quat q2) {
    Quat out;
    out.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    out.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    out.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    out.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    return out;
}

static inline void quat_normalize(Quat *q) {
    float n = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
    if (n > 0.0f) {
        q->w /= n;
        q->x /= n;
        q->y /= n;
        q->z /= n;
    }
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
    Quat qv = {0.0f, v.x, v.y, v.z};
    Quat tmp = quat_mul(q, qv);
    Quat q_conj = {q.w, -q.x, -q.y, -q.z};
    Quat res = quat_mul(tmp, q_conj);
    return (Vec3){res.x, res.y, res.z};
}

static inline Quat quat_inverse(Quat q) {
    return (Quat){q.w, -q.x, -q.y, -q.z};
}

static Quat rndquat(unsigned int* rng) {
    float u1 = rndf(0.0f, 1.0f, rng);
    float u2 = rndf(0.0f, 1.0f, rng);
    float u3 = rndf(0.0f, 1.0f, rng);

    float sqrt_1_minus_u1 = sqrtf(1.0f - u1);
    float sqrt_u1 = sqrtf(u1);

    float pi_2_u2 = 2.0f * (float)M_PI * u2;
    float pi_2_u3 = 2.0f * (float)M_PI * u3;

    Quat q;
    q.w = sqrt_1_minus_u1 * sinf(pi_2_u2);
    q.x = sqrt_1_minus_u1 * cosf(pi_2_u2);
    q.y = sqrt_u1 * sinf(pi_2_u3);
    q.z = sqrt_u1 * cosf(pi_2_u3);

    return q;
}

static void init_drone(CombatDrone* drone, float size, float dr, unsigned int* rng) {
    drone->params.arm_len = size / 2.0f;

    // m ~ x^3
    float mass_scale = powf(drone->params.arm_len, 3.0f) / powf(BASE_ARM_LEN, 3.0f);
    drone->params.mass = BASE_MASS * mass_scale * rndf(1.0f - dr, 1.0f + dr, rng);

    // I ~ mx^2
    float base_Iscale = BASE_MASS * BASE_ARM_LEN * BASE_ARM_LEN;
    float I_scale = drone->params.mass * powf(drone->params.arm_len, 2.0f) / base_Iscale;
    drone->params.ixx = BASE_IXX * I_scale * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.iyy = BASE_IYY * I_scale * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.izz = BASE_IZZ * I_scale * rndf(1.0f - dr, 1.0f + dr, rng);

    // k_thrust ~ m/l
    float k_thrust_scale = (drone->params.mass * drone->params.arm_len) / (BASE_MASS * BASE_ARM_LEN);
    drone->params.k_thrust = BASE_K_THRUST * k_thrust_scale * rndf(1.0f - dr, 1.0f + dr, rng);

    // k_ang_damp ~ I
    float base_avg_inertia = (BASE_IXX + BASE_IYY + BASE_IZZ) / 3.0f;
    float avg_inertia = (drone->params.ixx + drone->params.iyy + drone->params.izz) / 3.0f;
    float avg_inertia_scale = avg_inertia / base_avg_inertia;
    drone->params.k_ang_damp = BASE_K_ANG_DAMP * avg_inertia_scale * rndf(1.0f - dr, 1.0f + dr, rng);

    // drag ~ x^2
    float drag_scale = powf(drone->params.arm_len, 2.0f) / powf(BASE_ARM_LEN, 2.0f);
    drone->params.k_drag = BASE_K_DRAG * drag_scale * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.b_drag = BASE_B_DRAG * drag_scale * rndf(1.0f - dr, 1.0f + dr, rng);

    // Small gravity randomization
    drone->params.gravity = BASE_GRAVITY * rndf(0.99f, 1.01f, rng);

    // RPM ~ 1/x
    float rpm_scale = (BASE_ARM_LEN) / (drone->params.arm_len);
    drone->params.max_rpm = BASE_MAX_RPM * rpm_scale * rndf(1.0f - dr, 1.0f + dr, rng);

    drone->params.max_vel = BASE_MAX_VEL;
    drone->params.max_omega = BASE_MAX_OMEGA;

    drone->params.k_mot = BASE_K_MOT * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.j_mot = BASE_J_MOT * I_scale * rndf(1.0f - dr, 1.0f + dr, rng);

    // Initialize state
    for (int i = 0; i < 4; i++) {
        drone->state.rpms[i] = 0.0f;
    }
    drone->state.pos = (Vec3){0.0f, 0.0f, 0.0f};
    drone->prev_pos = drone->state.pos;
    drone->state.vel = (Vec3){0.0f, 0.0f, 0.0f};
    drone->state.omega = (Vec3){0.0f, 0.0f, 0.0f};
    drone->state.quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};

    // Initialize combat state
    for (int i = 0; i < MAX_BULLETS; i++) {
        drone->bullets[i].active = false;
    }
    drone->bullets_remaining = MAX_BULLETS;
    drone->disabled = false;
    drone->crashed = false;
    drone->hit_impact = 0.0f;
    drone->ground_height = 0.0f;
}

static void compute_derivatives(State* state, Params* params, float* actions, StateDerivative* derivatives) {
    // First order rpm lag
    float target_rpms[4];
    for (int i = 0; i < 4; i++) {
        target_rpms[i] = (actions[i] + 1.0f) * 0.5f * params->max_rpm;
    }

    // RPM rates
    float rpm_dot[4];
    for (int i = 0; i < 4; i++) {
        rpm_dot[i] = (1.0f / params->k_mot) * (target_rpms[i] - state->rpms[i]);
    }

    // Motor thrusts
    float T[4];
    for (int i = 0; i < 4; i++) {
        T[i] = params->k_thrust * powf(state->rpms[i], 2.0f);
    }

    // Body frame net force
    Vec3 F_prop_body = {0.0f, 0.0f, T[0] + T[1] + T[2] + T[3]};

    // Body frame force -> world frame force
    Vec3 F_prop = quat_rotate(state->quat, F_prop_body);

    // World frame linear drag
    Vec3 F_aero;
    F_aero.x = -params->b_drag * state->vel.x;
    F_aero.y = -params->b_drag * state->vel.y;
    F_aero.z = -params->b_drag * state->vel.z;

    // Velocity rates, a = F/m
    Vec3 v_dot;
    v_dot.x = (F_prop.x + F_aero.x) / params->mass;
    v_dot.y = (F_prop.y + F_aero.y) / params->mass;
    v_dot.z = ((F_prop.z + F_aero.z) / params->mass) - params->gravity;

    // Quaternion rates
    Quat omega_q = {0.0f, state->omega.x, state->omega.y, state->omega.z};
    Quat q_dot = quat_mul(state->quat, omega_q);
    q_dot.w *= 0.5f;
    q_dot.x *= 0.5f;
    q_dot.y *= 0.5f;
    q_dot.z *= 0.5f;

    // Body frame torques (cross copter)
    float arm_factor = params->arm_len / sqrtf(2.0f);
    Vec3 Tau_prop;
    Tau_prop.x = arm_factor * ((T[0] + T[1]) - (T[3] + T[2]));
    Tau_prop.y = arm_factor * ((T[0] + T[3]) - (T[1] + T[2]));
    Tau_prop.z = params->k_drag * (-T[0] + T[1] - T[2] + T[3]);

    // Torque from changing motor speeds
    float Tau_mot_z = params->j_mot * (rpm_dot[0] - rpm_dot[1] + rpm_dot[2] - rpm_dot[3]);

    // Torque from angular damping
    Vec3 Tau_aero;
    Tau_aero.x = -params->k_ang_damp * state->omega.x;
    Tau_aero.y = -params->k_ang_damp * state->omega.y;
    Tau_aero.z = -params->k_ang_damp * state->omega.z;

    // Gyroscopic torque
    Vec3 Tau_iner;
    Tau_iner.x = (params->iyy - params->izz) * state->omega.y * state->omega.z;
    Tau_iner.y = (params->izz - params->ixx) * state->omega.z * state->omega.x;
    Tau_iner.z = (params->ixx - params->iyy) * state->omega.x * state->omega.y;

    // Angular velocity rates
    Vec3 w_dot;
    w_dot.x = (Tau_prop.x + Tau_aero.x + Tau_iner.x) / params->ixx;
    w_dot.y = (Tau_prop.y + Tau_aero.y + Tau_iner.y) / params->iyy;
    w_dot.z = (Tau_prop.z + Tau_aero.z + Tau_iner.z + Tau_mot_z) / params->izz;

    derivatives->vel = state->vel;
    derivatives->v_dot = v_dot;
    derivatives->q_dot = q_dot;
    derivatives->w_dot = w_dot;
    for (int i = 0; i < 4; i++) {
        derivatives->rpm_dot[i] = rpm_dot[i];
    }
}

static void integrate_state(State* initial, StateDerivative* deriv, float dt, State* output) {
    output->pos = add3(initial->pos, scalmul3(deriv->vel, dt));
    output->vel = add3(initial->vel, scalmul3(deriv->v_dot, dt));
    output->quat = add_quat(initial->quat, scalmul_quat(deriv->q_dot, dt));
    output->omega = add3(initial->omega, scalmul3(deriv->w_dot, dt));
    for (int i = 0; i < 4; i++) {
        output->rpms[i] = initial->rpms[i] + deriv->rpm_dot[i] * dt;
    }
    quat_normalize(&output->quat);
}

static void rk4_step(State* state, Params* params, float* actions, float dt) {
    StateDerivative k1, k2, k3, k4;
    State temp_state;

    compute_derivatives(state, params, actions, &k1);

    integrate_state(state, &k1, dt * 0.5f, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k2);

    integrate_state(state, &k2, dt * 0.5f, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k3);

    integrate_state(state, &k3, dt, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k4);

    float dt_6 = dt / 6.0f;

    state->pos.x += (k1.vel.x + 2.0f * k2.vel.x + 2.0f * k3.vel.x + k4.vel.x) * dt_6;
    state->pos.y += (k1.vel.y + 2.0f * k2.vel.y + 2.0f * k3.vel.y + k4.vel.y) * dt_6;
    state->pos.z += (k1.vel.z + 2.0f * k2.vel.z + 2.0f * k3.vel.z + k4.vel.z) * dt_6;

    state->vel.x += (k1.v_dot.x + 2.0f * k2.v_dot.x + 2.0f * k3.v_dot.x + k4.v_dot.x) * dt_6;
    state->vel.y += (k1.v_dot.y + 2.0f * k2.v_dot.y + 2.0f * k3.v_dot.y + k4.v_dot.y) * dt_6;
    state->vel.z += (k1.v_dot.z + 2.0f * k2.v_dot.z + 2.0f * k3.v_dot.z + k4.v_dot.z) * dt_6;

    state->quat.w += (k1.q_dot.w + 2.0f * k2.q_dot.w + 2.0f * k3.q_dot.w + k4.q_dot.w) * dt_6;
    state->quat.x += (k1.q_dot.x + 2.0f * k2.q_dot.x + 2.0f * k3.q_dot.x + k4.q_dot.x) * dt_6;
    state->quat.y += (k1.q_dot.y + 2.0f * k2.q_dot.y + 2.0f * k3.q_dot.y + k4.q_dot.y) * dt_6;
    state->quat.z += (k1.q_dot.z + 2.0f * k2.q_dot.z + 2.0f * k3.q_dot.z + k4.q_dot.z) * dt_6;

    state->omega.x += (k1.w_dot.x + 2.0f * k2.w_dot.x + 2.0f * k3.w_dot.x + k4.w_dot.x) * dt_6;
    state->omega.y += (k1.w_dot.y + 2.0f * k2.w_dot.y + 2.0f * k3.w_dot.y + k4.w_dot.y) * dt_6;
    state->omega.z += (k1.w_dot.z + 2.0f * k2.w_dot.z + 2.0f * k3.w_dot.z + k4.w_dot.z) * dt_6;

    for (int i = 0; i < 4; i++) {
        state->rpms[i] += (k1.rpm_dot[i] + 2.0f * k2.rpm_dot[i] + 2.0f * k3.rpm_dot[i] + k4.rpm_dot[i]) * dt_6;
    }

    quat_normalize(&state->quat);
}

static void move_drone(CombatDrone* drone, float* actions, unsigned int* rng) {
    // If crashed, drone can't move
    if (drone->crashed) {
        return;
    }

    // Clamp actions
    clamp4(actions, -1.0f, 1.0f);

    // Domain randomized dt
    float dt = DT * rndf(1.0f - DT_RNG, 1.0f + DT_RNG, rng);

    // Update drone state
    drone->prev_pos = drone->state.pos;

    // If disabled, apply reduced control (can still move but impaired)
    if (drone->disabled) {
        // Reduce motor effectiveness by 50%
        float reduced_actions[4];
        for (int i = 0; i < 4; i++) {
            reduced_actions[i] = actions[i] * 0.5f;
        }
        rk4_step(&drone->state, &drone->params, reduced_actions, dt);
    } else {
        rk4_step(&drone->state, &drone->params, actions, dt);
    }

    // Clamp and normalize for observations
    clamp3(&drone->state.vel, -drone->params.max_vel, drone->params.max_vel);
    clamp3(&drone->state.omega, -drone->params.max_omega, drone->params.max_omega);

    // Check if disabled drone has crashed (hit ground)
    if (drone->disabled && drone->state.pos.z <= 0.0f) {
        drone->crashed = true;
        drone->ground_height = drone->state.pos.z;
        drone->state.pos.z = 0.0f;
        drone->state.vel = (Vec3){0.0f, 0.0f, 0.0f};
    }
}

static void fire_bullet(CombatDrone* drone, int owner_id) {
    if (drone->bullets_remaining <= 0) {
        return;
    }

    // Find an inactive bullet slot
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (!drone->bullets[i].active) {
            // Get forward direction from drone orientation
            Vec3 forward = {0.0f, 1.0f, 0.0f};  // Drone's forward in body frame
            Vec3 muzzle_dir = quat_rotate(drone->state.quat, forward);

            // Initialize bullet
            drone->bullets[i].pos = drone->state.pos;
            drone->bullets[i].vel = add3(drone->state.vel, scalmul3(muzzle_dir, BULLET_VELOCITY));
            drone->bullets[i].active = true;
            drone->bullets[i].owner = owner_id;

            drone->bullets_remaining--;
            drone->bullets_fired++;
            break;
        }
    }
}

static void update_bullets(CombatDrone* drone, float dt) {
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (!drone->bullets[i].active) {
            continue;
        }

        Bullet* bullet = &drone->bullets[i];

        // Update bullet position
        bullet->pos = add3(bullet->pos, scalmul3(bullet->vel, dt));

        // Apply drag and gravity
        Vec3 drag = scalmul3(bullet->vel, -BULLET_DRAG);
        Vec3 gravity = {0.0f, 0.0f, -BASE_GRAVITY};
        Vec3 accel = add3(drag, gravity);
        bullet->vel = add3(bullet->vel, scalmul3(accel, dt));

        // Deactivate if out of bounds or hit ground
        if (bullet->pos.x < -GRID_X || bullet->pos.x > GRID_X ||
            bullet->pos.y < -GRID_Y || bullet->pos.y > GRID_Y ||
            bullet->pos.z < 0.0f || bullet->pos.z > GRID_Z) {
            bullet->active = false;
        }
    }
}

static bool check_bullet_hit(Bullet* bullet, CombatDrone* target) {
    if (!bullet->active || target->crashed) {
        return false;
    }

    Vec3 to_target = sub3(bullet->pos, target->state.pos);
    float dist = norm3(to_target);

    return dist < HIT_RADIUS;
}

static void compute_observations(DroneWarsEnv *env) {
    for (int i = 0; i < env->num_agents; i++) {
        CombatDrone *drone = &env->drones[i];
        CombatDrone *opponent = &env->drones[1 - i];  // Other drone
        float *obs = env->agents[i].observations;
        int idx = 0;

        Quat q_inv = quat_inverse(drone->state.quat);
        Vec3 linear_vel_body = quat_rotate(q_inv, drone->state.vel);
        Vec3 to_opponent = sub3(opponent->state.pos, drone->state.pos);
        Vec3 opponent_vel_relative = sub3(opponent->state.vel, drone->state.vel);

        // Own drone state (14 features: 3 vel + 3 omega + 4 quat + 4 rpms)
        obs[idx++] = linear_vel_body.x / drone->params.max_vel;
        obs[idx++] = linear_vel_body.y / drone->params.max_vel;
        obs[idx++] = linear_vel_body.z / drone->params.max_vel;

        obs[idx++] = drone->state.omega.x / drone->params.max_omega;
        obs[idx++] = drone->state.omega.y / drone->params.max_omega;
        obs[idx++] = drone->state.omega.z / drone->params.max_omega;

        obs[idx++] = drone->state.quat.w;
        obs[idx++] = drone->state.quat.x;
        obs[idx++] = drone->state.quat.y;
        obs[idx++] = drone->state.quat.z;

        obs[idx++] = drone->state.rpms[0] / drone->params.max_rpm;
        obs[idx++] = drone->state.rpms[1] / drone->params.max_rpm;
        obs[idx++] = drone->state.rpms[2] / drone->params.max_rpm;
        obs[idx++] = drone->state.rpms[3] / drone->params.max_rpm;

        // Opponent information (9 features)
        obs[idx++] = clampf(to_opponent.x / GRID_X, -1.0f, 1.0f);
        obs[idx++] = clampf(to_opponent.y / GRID_Y, -1.0f, 1.0f);
        obs[idx++] = clampf(to_opponent.z / GRID_Z, -1.0f, 1.0f);

        obs[idx++] = clampf(opponent_vel_relative.x / BASE_MAX_VEL, -1.0f, 1.0f);
        obs[idx++] = clampf(opponent_vel_relative.y / BASE_MAX_VEL, -1.0f, 1.0f);
        obs[idx++] = clampf(opponent_vel_relative.z / BASE_MAX_VEL, -1.0f, 1.0f);

        obs[idx++] = opponent->disabled ? 1.0f : 0.0f;
        obs[idx++] = opponent->crashed ? 1.0f : 0.0f;
        obs[idx++] = (float)opponent->bullets_remaining / MAX_BULLETS;

        // Own combat status (2 features)
        obs[idx++] = (float)drone->bullets_remaining / MAX_BULLETS;
        obs[idx++] = drone->disabled ? 1.0f : 0.0f;

        // Incoming bullet positions (up to 3 bullets, 3 coords each = 9 features)
        int bullet_count = 0;
        for (int b = 0; b < MAX_BULLETS; b++) {
            if (opponent->bullets[b].active) {
                Vec3 to_bullet = sub3(opponent->bullets[b].pos, drone->state.pos);
                obs[idx++] = clampf(to_bullet.x / GRID_X, -1.0f, 1.0f);
                obs[idx++] = clampf(to_bullet.y / GRID_Y, -1.0f, 1.0f);
                obs[idx++] = clampf(to_bullet.z / GRID_Z, -1.0f, 1.0f);
                bullet_count++;
            }
        }
        // Pad remaining bullet slots with zeros
        for (int b = bullet_count; b < MAX_BULLETS; b++) {
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
        }
    }
}

static void reset_agent(DroneWarsEnv *env, CombatDrone *drone, int idx) {
    drone->episode_return = 0.0f;
    drone->episode_length = 0;
    drone->hits_scored = 0;
    drone->hits_received = 0;
    drone->bullets_fired = 0;

    float size = rndf(0.2f, 0.4f, &env->rng);
    init_drone(drone, size, 0.1f, &env->rng);

    // Position drones on opposite sides of the arena
    if (idx == 0) {
        drone->state.pos = (Vec3){-MARGIN_X / 2, 0.0f, GRID_Z / 2};
    } else {
        drone->state.pos = (Vec3){MARGIN_X / 2, 0.0f, GRID_Z / 2};
    }

    drone->prev_pos = drone->state.pos;
    drone->state.quat = rndquat(&env->rng);
}

static void add_log(DroneWarsEnv *env, int idx) {
    CombatDrone *drone = &env->drones[idx];
    CombatDrone *opponent = &env->drones[1 - idx];
    env->log.episode_return += drone->episode_return;
    env->log.episode_length += drone->episode_length;
    env->log.hits_scored += drone->hits_scored;
    env->log.hits_received += drone->hits_received;
    env->log.bullets_fired += drone->bullets_fired;

    if (drone->bullets_fired > 0) {
        env->log.accuracy += (float)drone->hits_scored / (float)drone->bullets_fired;
    }

    // Win: opponent is down (crashed or disabled) and this drone is not
    bool self_down = drone->crashed || drone->disabled;
    bool opp_down = opponent->crashed || opponent->disabled;
    float win = (opp_down && !self_down) ? 1.0f : 0.0f;
    env->log.perf += win;
    env->log.score += win;

    env->log.n += 1.0f;
}

void puf_reset(DroneWarsEnv *env) {
    env->tick = 0;

    for (int i = 0; i < env->num_agents; i++) {
        reset_agent(env, &env->drones[i], i);
        env->agents[i].rewards[0] = 0.0f;
        env->agents[i].terminals[0] = 0.0f;
    }

    compute_observations(env);
}

void puf_step(DroneWarsEnv *env) {
    env->tick = (env->tick + 1) % HORIZON;
    float rewards[NUM_DRONES] = {0};

    for (int i = 0; i < env->num_agents; i++) {
        CombatDrone *drone = &env->drones[i];

        // Actions: 4 motor controls + 1 fire button
        float *atn = env->agents[i].actions;
        float motor_actions[4] = {atn[0], atn[1], atn[2], atn[3]};
        bool fire = atn[4] > 0.0f;

        // Move drone
        move_drone(drone, motor_actions, &env->rng);

        // Fire bullet if action says so
        if (fire) {
            fire_bullet(drone, i);
        }

        // Update bullets
        update_bullets(drone, DT);

        drone->episode_length++;
    }

    // Check for bullet hits (after all drones have moved)
    for (int i = 0; i < env->num_agents; i++) {
        CombatDrone *shooter = &env->drones[i];
        CombatDrone *target = &env->drones[1 - i];

        for (int b = 0; b < MAX_BULLETS; b++) {
            if (check_bullet_hit(&shooter->bullets[b], target)) {
                // Hit detected!
                shooter->bullets[b].active = false;
                shooter->hits_scored++;
                target->hits_received++;

                // Calculate impact
                float relative_vel = norm3(sub3(shooter->bullets[b].vel, target->state.vel));
                target->hit_impact = relative_vel;

                // Award points for hit
                rewards[i] += 10.0f;
                rewards[1 - i] -= 5.0f;

                // Check if critical hit (immediate disable)
                if (relative_vel > CRITICAL_HIT_THRESHOLD) {
                    target->disabled = true;
                    rewards[i] += 20.0f;  // Bonus for critical hit
                }
            }
        }
    }

    // Check termination conditions
    bool episode_over = false;
    for (int i = 0; i < env->num_agents; i++) {
        CombatDrone *drone = &env->drones[i];
        CombatDrone *opponent = &env->drones[1 - i];

        // Out of bounds
        bool out_of_bounds =
            drone->state.pos.x < -GRID_X || drone->state.pos.x > GRID_X ||
            drone->state.pos.y < -GRID_Y || drone->state.pos.y > GRID_Y ||
            drone->state.pos.z < 0.0f || drone->state.pos.z > GRID_Z;

        if (out_of_bounds) {
            rewards[i] -= 10.0f;
            episode_over = true;
        }

        // Drone crashed
        if (drone->crashed) {
            rewards[1 - i] += 50.0f;  // Opponent wins
            episode_over = true;
        }

        // Timeout
        if (env->tick >= HORIZON - 1) {
            episode_over = true;

            // Award points based on health/status
            if (!drone->disabled && opponent->disabled) {
                rewards[i] += 30.0f;  // Win by superior condition
            } else if (drone->disabled && !opponent->disabled) {
                rewards[i] -= 30.0f;
            }
        }

        // Small survival reward
        if (!drone->crashed) {
            rewards[i] += 0.01f;
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        float r = clampf(rewards[i], -100.0f, 100.0f);
        env->drones[i].episode_return += r;
        env->agents[i].rewards[0] = r;
        // Both drones are reset together, so both episodes end together
        env->agents[i].terminals[0] = episode_over ? 1.0f : 0.0f;
    }

    if (episode_over) {
        for (int i = 0; i < env->num_agents; i++) {
            add_log(env, i);
        }
        env->tick = 0;
        for (int i = 0; i < env->num_agents; i++) {
            reset_agent(env, &env->drones[i], i);
        }
    }

    compute_observations(env);
}

// Helper to draw a drone as a simple 3D representation
static void draw_drone(CombatDrone* drone, Color color) {
    Vector3 pos = {drone->state.pos.x, drone->state.pos.z, drone->state.pos.y};  // Swap Y/Z for 3D view
    float arm_len = drone->params.arm_len * 10.0f;  // Scale for visibility

    // Draw drone body as a sphere
    DrawSphere(pos, arm_len * 0.5f, color);

    // Get drone orientation vectors
    Vec3 forward_body = {0.0f, 1.0f, 0.0f};
    Vec3 right_body = {1.0f, 0.0f, 0.0f};

    Vec3 forward = quat_rotate(drone->state.quat, forward_body);
    Vec3 right = quat_rotate(drone->state.quat, right_body);
    // Draw arms as lines (X configuration)
    float arm_scale = arm_len * 1.5f;
    Vector3 arm1_end = {pos.x + (forward.x + right.x) * arm_scale,
                        pos.y + (forward.z + right.z) * arm_scale,
                        pos.z + (forward.y + right.y) * arm_scale};
    Vector3 arm2_end = {pos.x + (forward.x - right.x) * arm_scale,
                        pos.y + (forward.z - right.z) * arm_scale,
                        pos.z + (forward.y - right.y) * arm_scale};
    Vector3 arm3_end = {pos.x + (-forward.x + right.x) * arm_scale,
                        pos.y + (-forward.z + right.z) * arm_scale,
                        pos.z + (-forward.y + right.y) * arm_scale};
    Vector3 arm4_end = {pos.x + (-forward.x - right.x) * arm_scale,
                        pos.y + (-forward.z - right.z) * arm_scale,
                        pos.z + (-forward.y - right.y) * arm_scale};

    DrawLine3D(pos, arm1_end, color);
    DrawLine3D(pos, arm2_end, color);
    DrawLine3D(pos, arm3_end, color);
    DrawLine3D(pos, arm4_end, color);

    // Draw forward direction indicator
    Vector3 nose = {pos.x + forward.x * arm_len * 2.0f,
                    pos.y + forward.z * arm_len * 2.0f,
                    pos.z + forward.y * arm_len * 2.0f};
    DrawLine3D(pos, nose, WHITE);

    // Draw status indicator
    if (drone->disabled) {
        DrawSphere(pos, arm_len * 0.3f, ORANGE);
    }
    if (drone->crashed) {
        DrawSphere(pos, arm_len * 0.3f, DARKGRAY);
    }
}

// Draw active bullets
static void draw_bullets(CombatDrone* drone, Color color) {
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (drone->bullets[i].active) {
            Vector3 bullet_pos = {drone->bullets[i].pos.x,
                                  drone->bullets[i].pos.z,
                                  drone->bullets[i].pos.y};
            DrawSphere(bullet_pos, 0.2f, color);
        }
    }
}

static void update_camera(Client* client) {
    float yaw_rad = client->camera_yaw * (float)M_PI / 180.0f;
    float pitch_rad = client->camera_pitch * (float)M_PI / 180.0f;
    float cos_pitch = cosf(pitch_rad);
    client->camera.position = (Vector3){
        CAMERA_DISTANCE * cos_pitch * sinf(yaw_rad),
        CAMERA_DISTANCE * cos_pitch * cosf(yaw_rad),
        CAMERA_DISTANCE * sinf(pitch_rad) + client->camera.target.z
    };
}

void puf_render(DroneWarsEnv* env) {
    // Lazy initialization of raylib window
    if (env->client == NULL) {
        if (!IsWindowReady()) {
            InitWindow(WIDTH, HEIGHT, "PufferLib Drone Wars");
            SetTargetFPS(60);
        }
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->width = WIDTH;
        env->client->height = HEIGHT;
        env->client->show_trails = true;
        env->client->show_bullets = true;
        env->client->paused = false;

        env->client->camera_yaw = 0.0f;    // start looking from +Y direction
        env->client->camera_pitch = 45.0f; // 45 degree angle from above

        env->client->camera.target = (Vector3){0.0f, 0.0f, 15.0f};
        env->client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
        env->client->camera.fovy = 45.0f;
        env->client->camera.projection = CAMERA_PERSPECTIVE;
        update_camera(env->client);
    }

    if (IsKeyPressed(KEY_ESCAPE)) {
        exit(0);
    }

    // Camera control with arrow keys
    Client* client = env->client;
    bool camera_changed = false;
    if (IsKeyDown(KEY_LEFT)) {
        client->camera_yaw -= CAMERA_ANGLE_SPEED;
        camera_changed = true;
    }
    if (IsKeyDown(KEY_RIGHT)) {
        client->camera_yaw += CAMERA_ANGLE_SPEED;
        camera_changed = true;
    }
    if (IsKeyDown(KEY_UP)) {
        client->camera_pitch = clampf(client->camera_pitch + CAMERA_ANGLE_SPEED,
            CAMERA_MIN_PITCH, CAMERA_MAX_PITCH);
        camera_changed = true;
    }
    if (IsKeyDown(KEY_DOWN)) {
        client->camera_pitch = clampf(client->camera_pitch - CAMERA_ANGLE_SPEED,
            CAMERA_MIN_PITCH, CAMERA_MAX_PITCH);
        camera_changed = true;
    }
    if (camera_changed) {
        update_camera(client);
    }

    BeginDrawing();
    ClearBackground(DARKGRAY);

    BeginMode3D(client->camera);

    // Draw ground plane
    DrawPlane((Vector3){0, 0, 0}, (Vector2){GRID_X * 2, GRID_Y * 2}, (Color){50, 50, 50, 255});

    // Draw arena boundaries
    DrawCubeWires((Vector3){0, 0, GRID_Z / 2}, GRID_X * 2, GRID_Y * 2, GRID_Z, GREEN);

    draw_drone(&env->drones[0], RED);
    draw_drone(&env->drones[1], BLUE);
    draw_bullets(&env->drones[0], YELLOW);
    draw_bullets(&env->drones[1], ORANGE);

    EndMode3D();

    // Draw UI overlay
    DrawText(TextFormat("Tick: %d", env->tick), 10, 10, 20, WHITE);
    DrawText(TextFormat("Drone 1: Bullets=%d Disabled=%d",
                       env->drones[0].bullets_remaining, env->drones[0].disabled),
             10, 40, 18, RED);
    DrawText(TextFormat("Drone 2: Bullets=%d Disabled=%d",
                       env->drones[1].bullets_remaining, env->drones[1].disabled),
             10, 65, 18, BLUE);
    DrawText("[Arrow Keys] Rotate Camera  [ESC] Exit", 10, HEIGHT - 30, 16, GRAY);

    EndDrawing();
    puf_web_vsync();
}

void puf_close(DroneWarsEnv *env) {
    if (env->client != NULL) {
        free(env->client);
        env->client = NULL;
        if (IsWindowReady()) {
            CloseWindow();
        }
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = NUM_DRONES;  // 1v1 combat
    env->tick = 0;
    env->client = NULL;
    env->log = (Log){0};
    for (int i = 0; i < NUM_DRONES; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "hits_scored", log->hits_scored);
    dict_set(out, "hits_received", log->hits_received);
    dict_set(out, "bullets_fired", log->bullets_fired);
    dict_set(out, "accuracy", log->accuracy);
    dict_set(out, "n", log->n);
}
