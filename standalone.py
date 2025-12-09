import time
import threading
import copy
import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
import pinocchio as pin
from pinocchio import FrameType, ReferenceFrame
import viser
from viser.extras import ViserUrdf
from casadi import (
    SX,
    DM,
    dot,
    mtimes,
    reshape,
    log,
    sum1,
    nlpsol,
    vertcat,
)
from scipy.linalg import block_diag
from scipy import sparse
import osqp
from yourdfpy import URDF
import select
import sys
import tty
import termios

# =============================================================================
# 1. Configuration and Constants
# =============================================================================

urdf_path = "assets/scene/robotis_5F_hand/default.urdf"
mesh_path = "assets/scene/robotis_5F_hand"
env_path = "assets/scene/robotis_5F_hand/default.xml"

default_q = [
  -0.048, 
  -0.983, -1.71, 0.619, 0.523, 
  -0.6, 0.8, 0.6, 0.6, 
  -0.1, 0.6, 0.6, 0.6, 
  0.1, 0.6, 0.6, 0.6, 
  0.6, 0.8, 0.6, 0.6]

kp = np.array([500.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
kd = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
n_joints = len(default_q)
table_height = -0.2 

K_a = [1.0 for _ in range(n_joints)]
K_a[0] = 100.0
K_a = np.array(K_a, dtype=np.float64)

M_o = np.array([0.01, 0.01, 0.01], dtype=np.float64)
eps = 0.1 
h = 0.1 
mu = 0.1 
kappa = 10000.0 

# =============================================================================
# 2. Math Utilities
# =============================================================================

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def batch_skew(v):
    # v shape: (N, 3)
    N = v.shape[0]
    z = np.zeros(N)
    return np.array([
        [z, -v[:, 2], v[:, 1]],
        [v[:, 2], z, -v[:, 0]],
        [-v[:, 1], v[:, 0], z]
    ]).transpose(2, 0, 1)

# =============================================================================
# 3. Pinocchio Kinematics Wrappers
# =============================================================================

def get_fk_link_poses(q, pin_model, pin_data, link_names): 
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)

    link_poses = [] 
    for link_name in link_names: 
        ee_frame_id = pin_model.getFrameId(link_name)
        link_pose = pin_data.oMf[ee_frame_id]
        link_pose_mat = np.eye(4)
        link_pose_mat[:3, :3] = link_pose.rotation
        link_pose_mat[:3, 3] = link_pose.translation
        link_poses.append(copy.deepcopy(link_pose_mat))
    
    return copy.deepcopy(link_poses)

def get_link_jacobians(q, pin_model, pin_data, link_names): 
    pin.computeJointJacobians(pin_model, pin_data, q)
    J_list = []

    for link_name in link_names: 
        ee_frame_id = pin_model.getFrameId(link_name)
        J = pin.getFrameJacobian(pin_model, pin_data, ee_frame_id, ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = np.vstack((J[3:, :], J[:3, :])) # Swap to linear/angular
        J_list.append(copy.deepcopy(J))

    return copy.deepcopy(J_list)

def get_link_dJacobians_dq(q, pin_model, pin_data, link_names):
    nv = pin_model.nv
    n_links = len(link_names)
    pin.computeJointJacobians(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    frame_ids = [pin_model.getFrameId(name) for name in link_names]
    dJdq = np.zeros((n_links, 6, nv, nv))
    v = np.zeros(nv)

    for i in range(nv):
        v[:] = 0.0
        v[i] = 1.0
        pin.computeJointJacobiansTimeVariation(pin_model, pin_data, q, v)
        for ell, fid in enumerate(frame_ids):
            dJ = pin.getFrameJacobianTimeVariation(
                pin_model, pin_data, fid, ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            dJdq[ell, :, :, i] = dJ

    dJdq = np.concatenate((dJdq[:, 3:, :, :], dJdq[:, :3, :, :]), axis=1)
    return dJdq

# =============================================================================
# 4. Contact Physics and Geometry
# =============================================================================

def compute_contact_spheres(link_poses, link_jacobians, all_links_dJacobians_dq):
    contact_sphere_params = {}
    
    # Define sphere offsets relative to links
    # Indices correspond to list_link_names order in main loop
    # 2: thumb4, 4: index4, 6: middle4, 8: ring4, 10: little4
    
    thumb4_pose = link_poses[2]
    contact_sphere_params['thumb_1'] = {"center": (thumb4_pose[:3, 3] + 0.027*thumb4_pose[:3, 2] - 0.001*thumb4_pose[:3, 0]), "radius": 0.015} 

    index4_pose = link_poses[4]
    contact_sphere_params['index_1'] = {"center": (index4_pose[:3, 3] + 0.027*index4_pose[:3, 2] - 0.001*index4_pose[:3, 0]), "radius": 0.015} 

    middle4_pose = link_poses[6]
    contact_sphere_params['middle_1'] = {"center": (middle4_pose[:3, 3] + 0.027*middle4_pose[:3, 2] - 0.001*middle4_pose[:3, 0]), "radius": 0.015}

    ring4_pose = link_poses[8]
    contact_sphere_params['ring_1'] = {"center": (ring4_pose[:3, 3] + 0.027*ring4_pose[:3, 2] - 0.001*ring4_pose[:3, 0]), "radius": 0.015}

    little4_pose = link_poses[10]
    contact_sphere_params['little_1'] = {"center": (little4_pose[:3, 3] + 0.027*little4_pose[:3, 2] - 0.001*little4_pose[:3, 0]), "radius": 0.015}

    contact_sphere_vec = [] 
    for key in contact_sphere_params.keys(): 
        contact_sphere_vec.append(np.hstack((contact_sphere_params[key]['center'], contact_sphere_params[key]['radius'])))
    contact_sphere_vec = np.array(contact_sphere_vec)

    link_J_for_spheres = np.concatenate([link_jacobians[2], link_jacobians[4], link_jacobians[6], link_jacobians[8], link_jacobians[10]])
    link_J_for_spheres = link_J_for_spheres.reshape(len(contact_sphere_params), 6, n_joints)

    link_centers_for_sphere = np.concatenate([
        thumb4_pose[:3, 3].reshape(1, 3),
        index4_pose[:3, 3].reshape(1, 3),
        middle4_pose[:3, 3].reshape(1, 3),
        ring4_pose[:3, 3].reshape(1, 3),
        little4_pose[:3, 3].reshape(1, 3),
    ])

    m = n_joints 
    dlink_jacobians_cat_dq = np.concatenate([
        all_links_dJacobians_dq[2].reshape(1, 6, m, m),
        all_links_dJacobians_dq[4].reshape(1, 6, m, m),
        all_links_dJacobians_dq[6].reshape(1, 6, m, m),
        all_links_dJacobians_dq[8].reshape(1, 6, m, m),
        all_links_dJacobians_dq[10].reshape(1, 6, m, m),
    ])

    return contact_sphere_vec, link_J_for_spheres, link_centers_for_sphere, contact_sphere_params, dlink_jacobians_cat_dq

def compute_contact_states(contact_sphere_vec, link_centers_for_sphere, link_J_for_spheres, q_o, object_radius, table_height, threshold_phi=0.001, dlink_jacobians_cat_dq=None):
    N = contact_sphere_vec.shape[0]
    n_control = 21 
    n = n_control + 3 

    robot_sphere_center = contact_sphere_vec[:, :3]
    robot_sphere_radius = contact_sphere_vec[:, 3]
    object_sphere_center = q_o.reshape(1, 3)
    object_sphere_radius = object_radius

    alpha = object_sphere_center - robot_sphere_center
    alpha_norm = np.linalg.norm(alpha, axis=1).reshape(N, 1)
    alpha_hat = alpha / (alpha_norm + 1e-8)

    phi = np.zeros((N+1,), dtype=np.float64)
    phi[0] = object_sphere_center[0, 2] - table_height  - object_sphere_radius
    phi[1:] = alpha_norm.flatten() - (robot_sphere_radius + object_sphere_radius)

    contact_flag = phi < threshold_phi

    contact_pos = np.zeros((N+1, 3), dtype=np.float64)
    contact_pos[0, :] = object_sphere_center[0, :] - np.array([0.0, 0.0, object_sphere_radius])
    contact_pos[1:, :] = robot_sphere_center + (robot_sphere_radius.reshape((-1, 1)).repeat(3, axis=1) * alpha_hat)

    n_hat = np.zeros((N+1, 1, 3), dtype=np.float64)
    n_hat[0, 0, :] = np.array([0, 0, 1])
    n_hat[1:, 0, :] = alpha_hat 
    n_hat_partial = n_hat[1:, 0, :]

    t_hat = np.zeros((N+1, 2, 3), dtype=np.float64)
    t_hat[0, 0, :] = np.array([1, 0, 0])
    t_hat[0, 1, :] = np.array([0, 1, 0])

    J_o = np.zeros((N+1, 3, 6), dtype=np.float64)
    J_o[0, :, 3:] = np.eye(3)
    J_o[0, :, :3] = -skew(np.array([0, 0, -1.0]))
    batch_r = contact_pos[1:] - object_sphere_center
    J_o[1:, :, 3:] = np.eye(3).reshape(1, 3, 3).repeat(N, axis=0)
    J_o[1:, :, :3] = -batch_skew(batch_r)

    r_vector = np.ones((N, 3))
    t_vector = r_vector - n_hat_partial * (r_vector * n_hat_partial).sum(axis=1).reshape(-1, 1)
    t_vector_norm = np.linalg.norm(t_vector, axis=1).reshape(-1, 1)
    t_hat[1:, 0, :] = t_vector / t_vector_norm
    t_hat[1:, 1, :] = np.cross(n_hat_partial, t_hat[1:, 0, :])

    J_o = J_o[:, :, 3:]
    J_a = np.zeros((N+1, 3, n_control), dtype=np.float64)

    delta = contact_pos[1:] - link_centers_for_sphere
    T = np.zeros((N, 3, 6), dtype=np.float64)
    T[:, :, :3] = -batch_skew(delta)
    T[:, :, 3:] = np.eye(3).reshape(1, 3, 3).repeat(N, axis=0)
    J_a[1:] = T @ link_J_for_spheres

    J_c_temp = np.concatenate([J_o, -J_a], axis=-1)
    J_cn = n_hat @ J_c_temp
    J_ct = t_hat @ J_c_temp
    J_c = np.concatenate([J_cn, J_ct], axis=1)

    # Derivatives
    delta2 = robot_sphere_center - link_centers_for_sphere
    transformation_matrix2 = np.zeros((N, 3, 6), dtype=np.float64)
    transformation_matrix2[:, :3, :3] = -batch_skew(delta2)
    transformation_matrix2[:, :3, 3:6] = np.eye(3).reshape(1, 3, 3).repeat(N, axis=0)
    drobot_sphere_center_dq = transformation_matrix2 @ link_J_for_spheres

    dn_hat_dq = np.zeros((N + 1, 1, 3, n), dtype=np.float64)
    temp = -alpha.reshape(N, 3, 1) @ alpha.reshape(N, 1, 3)
    temp = temp / (alpha_norm.reshape(N, 1, 1) ** 2)
    temp = temp + np.eye(3).reshape(1, 3, 3).repeat(N, axis=0)
    temp = temp / alpha_norm.reshape(N, 1, 1)
    dn_hat_dq[1:, 0, :, :3] = temp
    dn_hat_dq[1:, 0, :, 3:] = -temp @ drobot_sphere_center_dq

    dcontact_pos_dq = np.zeros((N, 3, n), dtype=np.float64)
    dcontact_pos_dq[:, :, :3] += 0.5 * np.eye(3).reshape(1, 3, 3).repeat(N, axis=0)
    dcontact_pos_dq[:, :, 3:] += 0.5 * drobot_sphere_center_dq
    dn_hat_dq_partial = dn_hat_dq[1:, 0, :, :]
    dcontact_pos_dq -= 0.5 * (object_radius - robot_sphere_radius).reshape(-1, 1, 1) * dn_hat_dq_partial

    dlink_centers_cat_dq = np.zeros((N, 3, n), dtype=np.float64)
    dlink_centers_cat_dq[:, :, 3:] = link_J_for_spheres[:, 3:, :]
    ddelta_dq = dcontact_pos_dq - dlink_centers_cat_dq

    dtransformation_matrix_dq = np.zeros((N, 3, 6, n), dtype=np.float64)
    temp = -batch_skew(np.transpose(ddelta_dq, (0, 2, 1)).reshape(-1, 3))
    dtransformation_matrix_dq[:, :3, :3, :] = np.transpose(temp.reshape(N, n, 3, 3), (0, 2, 3, 1))

    dt_hat_dq = np.zeros((N + 1, 2, 3, n), dtype=np.float64)
    temp = -t_vector.reshape(N, 3, 1) @ t_vector.reshape(N, 1, 3)
    temp = temp / (t_vector_norm.reshape(N, 1, 1) ** 2)
    temp = temp + np.eye(3).reshape(1, 3, 3).repeat(N, axis=0)
    temp = temp / t_vector_norm.reshape(N, 1, 1)
    dndq_rT_n = np.einsum("nij, nk, nk -> nij", dn_hat_dq_partial, r_vector, n_hat_partial)
    n_rT_dndq = np.einsum("ni, nj, njk -> nik", n_hat_partial, r_vector, dn_hat_dq_partial)
    dt_hat_dq[1:, 0, :, :] = temp @ (-(dndq_rT_n + n_rT_dndq))

    term1 = -batch_skew(t_hat[1:, 0, :]) @ dn_hat_dq[1:, 0, :, :]
    term2 = batch_skew(n_hat[1:, 0, :]) @ dt_hat_dq[1:, 0, :, :]
    dt_hat_dq[1:, 1, :, :] = term1 + term2

    dphi_dq = np.zeros((N + 1, n), dtype=np.float64)
    dphi_dq[0, :3] = np.array([0.0, 0.0, 1.0])
    dphi_dq[1:, :3] = alpha_hat
    dphi_dq[1:, 3:] = (-dphi_dq[1:, :3].reshape(N, 3, 1) * drobot_sphere_center_dq).sum(axis=1)

    m = n_joints
    dJ_o_dq = np.zeros((N + 1, 3, 3, n), dtype=np.float64)
    dJ_a_dq = np.zeros((N + 1, 3, m, n), dtype=np.float64)
    dJ_a_dq[1:, :, :, :] = np.einsum("nijk, njl -> nilk", dtransformation_matrix_dq, link_J_for_spheres)
    dJ_a_dq[1:, :, :, 3:] += np.einsum("nij, njkl -> nikl", T, dlink_jacobians_cat_dq)
    dJ_c_temp_dq = np.concatenate([dJ_o_dq, -dJ_a_dq], axis=-2)
    d1J_cn_dq = np.einsum("nijk, njl -> nilk", dn_hat_dq, J_c_temp)
    d1J_ct_dq = np.einsum("nijk, njl -> nilk", dt_hat_dq, J_c_temp)
    d2J_cn_dq = np.einsum("nij, njkl -> nikl", n_hat, dJ_c_temp_dq)
    d2J_ct_dq = np.einsum("nij, njkl -> nikl", t_hat, dJ_c_temp_dq)

    dJ_cn_dq = d1J_cn_dq + d2J_cn_dq
    dJ_ct_dq = d1J_ct_dq + d2J_ct_dq
    dJ_c_dq = np.concatenate([dJ_cn_dq, dJ_ct_dq], axis=1)

    return contact_pos, phi, contact_flag, n_hat, t_hat, J_c, dphi_dq, dJ_c_dq

# =============================================================================
# 5. Quasidynamic Dynamics and Optimization
# =============================================================================

def compute_kinodynamic_matrices(q_o, q_a, u, contact_pos, phi, contact_flag, J_c):
    contact_pos = contact_pos[contact_flag] 
    phi = phi[contact_flag]
    J_c = J_c[contact_flag]

    P_diag = np.zeros(len(q_a) + len(q_o), dtype=np.float64)
    P_diag[:len(q_o)] = eps * M_o / h**2 
    P_diag[len(q_o):] = K_a

    b = np.zeros(len(q_a) + len(q_o), dtype=np.float64)
    tau_o = -M_o * np.array([0, 0, 9.81])
    b[:len(q_o)] = -((eps * M_o * q_o / h**2) + tau_o)
    tau_a = np.zeros((len(q_a),), dtype=np.float64)
    b[len(q_o):] = -(K_a * u + tau_a)

    c = -J_c @ np.hstack([q_o, q_a])
    c[:, 0] += phi 

    return P_diag, b, J_c, c 

def forward_dynamics(q_o, q_a, J_c_active, P_diag, b, c):
    init_guess = np.hstack([q_o, q_a])
    n = P_diag.shape[0]
    nc = J_c_active.shape[0]

    x = SX.sym("x", n)
    J_c_dm = DM(J_c_active.reshape(-1, n))
    P_diag_dm = DM(P_diag.reshape(-1, 1))
    b_dm = DM(b.reshape(-1, 1))
    c_dm = DM(c.reshape(-1, 1))

    quad = 0.5 * dot(P_diag_dm, x**2)
    lin = dot(b_dm, x)
    nu_flat = mtimes(J_c_dm, x) + c_dm
    nu = reshape(nu_flat, 3, nc).T
    nu_n = nu[:, 0]
    nu_tx = nu[:, 1]
    nu_ty = nu[:, 2]
    nu_t_sq = nu_tx**2 + nu_ty**2
    arg = nu_n**2 / (mu**2) - nu_t_sq
    log_barrier = -(1.0/kappa) * sum1(log(arg))
    f = quad + lin + log_barrier 
    g = vertcat(nu_n, arg)

    nlp = {"x": x, "f": f, "g": g}
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes",
        "ipopt":{
            "nlp_scaling_method": "gradient-based",
            "tol": 1e-8, 
            "warm_start_bound_push": 1e-10,
        }
    }

    solver = nlpsol("solver", "ipopt", nlp, opts)
    x0 = init_guess 
    lbg = DM.zeros(g.shape[0])
    ubg = DM.inf(g.shape[0])

    try: 
        sol = solver(x0=x0, lbg=lbg, ubg=ubg)
        if solver.stats()["success"]:
            q_plus = sol["x"].toarray()
        else:
            q_plus = init_guess.reshape(-1, 1)
    except Exception:
        q_plus = init_guess.reshape(-1, 1)

    def compute_lambda(q_plus, J_c_active, c, kappa, mu):
        nu = (J_c_active @ q_plus).reshape(-1, 3) + c
        nu_n = nu[:, 0]
        nu_t = nu[:, 1:]
        nu_t_sq = nu_t**2 
        arg = nu_n**2 / (mu**2) - nu_t_sq.sum(axis=-1)
        coeff = 2*1/kappa * (1/arg).reshape(-1, 1)
        lambda_ = coeff * np.concatenate([nu_n.reshape(-1, 1)/mu**2, -nu_t], axis=-1)
        return lambda_

    lambda_plus = compute_lambda(q_plus, J_c_active, c, kappa, mu)
    return q_plus.reshape(-1), lambda_plus

# =============================================================================
# 6. Sensitivity Analysis
# =============================================================================

def solve_linear_with_schur_complement(A_diag, B, C, D_diag, b):
    n = A_diag.shape[0]
    m = D_diag.shape[0]
    k = b.shape[1]
    x_sol = np.zeros((n + m, k))
    inv_A = np.diag(1 / A_diag)
    inv_D = np.diag(1 / D_diag)
    
    if n >= m:
        S = np.diag(D_diag) - C @ inv_A @ B
        beta = b[n:, :] - C @ inv_A @ b[:n, :]
        alpha = np.linalg.solve(S, beta)
        x_sol[n:, :] = alpha
        x_sol[:n, :] = inv_A @ (b[:n, :] - B @ alpha)
    else:
        S = np.diag(A_diag) - B @ inv_D @ C
        beta = b[:n, :] - B @ inv_D @ b[n:, :]
        alpha = np.linalg.solve(S, beta)
        x_sol[:n, :] = alpha
        x_sol[n:, :] = inv_D @ (b[n:, :] - C @ alpha)
    return x_sol

def compute_sensitivity(P_diag, J, lambda_, q, q_plus, c, dbdu, dbdq, dphidq, dJdq, kappa, mu):
    n = q_plus.shape[0]
    nc = J.shape[0]
    m = dbdu.shape[1]

    dcdq = -np.einsum("nijk, j -> nik", dJdq, q) - J
    dcdq[:, 0, :] += dphidq
    nu = np.einsum("nij, j -> ni", J, q_plus) + c
    partial_q_nu = np.einsum("nijk, j -> nik", dJdq, q_plus) + dcdq

    J_flatten = J.reshape(-1, n)
    partial_q_nu_flatten = partial_q_nu.reshape(-1, n)

    A = np.array([1.0 / mu**2, -1.0, -1.0], dtype=nu.dtype)
    lambda_nuT_A = np.einsum("ni, nj, j -> nij", lambda_, nu, A)
    lambda_nuT_A = block_diag(*lambda_nuT_A)
    lambda_nuT_A_minus_1_kappa_A = lambda_nuT_A - 1 / kappa * block_diag(*np.diag(A).reshape(1, 3, 3).repeat(nc, axis=0))

    A11_diag = P_diag
    A12 = -J_flatten.T
    A21 = lambda_nuT_A_minus_1_kappa_A @ J_flatten
    A22_diag = 0.5 * np.einsum("ni, ni, i -> n", nu, nu, A).reshape(-1, 1).repeat(3, axis=1).reshape(-1)

    B = np.zeros((n + 3 * nc, n + m))
    B[:n, :n] = -dbdq + np.einsum("nijk, ni -> jk", dJdq, lambda_)
    B[:n, n:] = -dbdu
    B[n:, :n] = -lambda_nuT_A_minus_1_kappa_A @ partial_q_nu_flatten

    x_sol = solve_linear_with_schur_complement(A11_diag, A12, A21, A22_diag, B)

    dq_plus_dq = x_sol[:n, :n]
    dlambda_dq = x_sol[n:, :n].reshape(nc, 3, n)
    dq_plus_du = x_sol[:n, n:]
    dlambda_du = x_sol[n:, n:].reshape(nc, 3, m)
    return dq_plus_dq, dlambda_dq, dq_plus_du, dlambda_du

# =============================================================================
# 7. Task Planner and MPC Controller
# =============================================================================

# Functions removed and inlined into main loop

# =============================================================================
# 8. Visualization Setup
# =============================================================================

viser_server = viser.ViserServer()     
urdf = URDF.load(urdf_path, mesh_dir=mesh_path, load_meshes=True, build_scene_graph=True, load_collision_meshes=False, build_collision_scene_graph=False)
viser_robot = ViserUrdf(viser_server, urdf_or_path=urdf, load_meshes=True, load_collision_meshes=False, mesh_color_override=(0.6, 0.6, 0.6, 1.0), collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5))
viser_server.scene.add_box(name="/table", dimensions=(0.9, 1.4, 0.08), color=(0.76, 0.60, 0.42), position=(0, 0, table_height - 0.08 / 2), receive_shadow=True, cast_shadow=True)
viser_server.scene.add_light_directional(name="/control0/directional_light", color=(186, 219, 173), cast_shadow=True)
viser_server.scene.add_frame(name="/WorldAxes", axes_length=0.1, axes_radius=0.0015, origin_radius=0.002)
viser_robot.update_cfg(default_q)

vis_spheres = {} 
def visualize_spheres(contact_sphere_params):
    for key, contact_sphere in contact_sphere_params.items():
        if key in vis_spheres:
            sphere = vis_spheres[key]
            sphere.position = contact_sphere['center']
            sphere.radius = contact_sphere['radius']
        else:
            vis_spheres[key]  = viser_server.scene.add_icosphere(
                name=key, position=contact_sphere['center'], radius=contact_sphere['radius'],
                color=contact_sphere.get('color', (1.0, 0.0, 0.0)), cast_shadow=True, receive_shadow=True, opacity=0.5
            )

def visualize_contact_force(contact_pos, contact_forces):
    points = []
    for i, pos in enumerate(contact_pos):
        force = contact_forces[i]
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > 1e-6:
            force_dir = force / force_magnitude
            arrow_length = min(force_magnitude * 0.2, 0.1)
            points.append([pos, pos + force_dir * arrow_length])
    if len(points) > 0:
        viser_server.scene.add_line_segments(name=f"/contact_forces", points=points, line_width=4, colors=(0.0, 0.0, 1.0))

# =============================================================================
# 9. Threaded MuJoCo Simulation Class
# =============================================================================

class MuJoCoSimThread(threading.Thread):
    def __init__(self, model_path):
        super().__init__()
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        self.running = True
        self.lock = threading.Lock()
        
        # Initialize state
        self.data.qpos[:n_joints] = default_q
        self.ctrl_command = np.array(default_q)
        
        # Store default object position
        self.q_o_default = self.data.qpos[n_joints:n_joints+3].copy()
        
        # Shared state for controller to read
        self.q_a = self.data.qpos[:n_joints].copy()
        self.q_ad = self.data.qvel[:n_joints].copy()
        self.q_o = self.data.qpos[n_joints:n_joints+3].copy()
        self.object_radius = self.model.geom_size[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "object_geom")][0]
        self.viewer = None

    def reset_state(self):
        with self.lock:
            # Reset robot
            self.data.qpos[:n_joints] = default_q
            self.data.qvel[:n_joints] = 0
            
            # Reset object
            self.data.qpos[n_joints:n_joints+3] = self.q_o_default
            # Reset object velocity (linear and angular if exists)
            self.data.qvel[n_joints:] = 0
            
            self.ctrl_command = np.array(default_q)
            
            # Update shared state immediately
            self.q_a = self.data.qpos[:n_joints].copy()
            self.q_ad = self.data.qvel[:n_joints].copy()
            self.q_o = self.data.qpos[n_joints:n_joints+3].copy()

    def run(self):
        # Launch the passive viewer
        self.viewer = mjv.launch_passive(self.model, self.data)
        
        while self.running and self.viewer.is_running():
            start_time = time.time()
            with self.lock:
                # Apply PD control
                q_current = self.data.qpos[:n_joints]
                q_vel = self.data.qvel[:n_joints]
                tau = kp * (self.ctrl_command - q_current) + kd * (-q_vel)
                self.data.ctrl[:n_joints] = tau
                
                mj.mj_step(self.model, self.data)
                
                # Update shared state
                self.q_a = self.data.qpos[:n_joints].copy()
                self.q_ad = self.data.qvel[:n_joints].copy()
                self.q_o = self.data.qpos[n_joints:n_joints+3].copy()
            
            # Sync the viewer
            self.viewer.sync()

            # Sleep to maintain roughly real-time (optional, or run as fast as possible)
            elapsed = time.time() - start_time
            if elapsed < self.model.opt.timestep:
                time.sleep(self.model.opt.timestep - elapsed)
        
        if self.viewer.is_running():
            self.viewer.close()
        self.running = False

    def get_state(self):
        with self.lock:
            return self.q_a.copy(), self.q_ad.copy(), self.q_o.copy(), self.object_radius

    def set_control(self, u):
        with self.lock:
            self.ctrl_command = u.copy()

    def stop(self):
        self.running = False

# =============================================================================
# 10. Main Control Loop
# =============================================================================

if __name__ == "__main__":
    # Setup Pinocchio
    pin_robot = pin.RobotWrapper.BuildFromURDF(urdf_path, [mesh_path])
    pin_model = pin_robot.model 
    pin_data = pin_robot.data
    list_link_names = [
        "palm", "finger_r_link_1_thumb3", "finger_r_link_1_thumb4",
        "finger_r_link_2_index3", "finger_r_link_2_index4",
        "finger_r_link_3_middle3", "finger_r_link_3_middle4",
        "finger_r_link_4_ring3", "finger_r_link_4_ring4",
        "finger_r_link_5_little3", "finger_r_link_5_little4",
    ]

    # Controller State
    controller_mode = "reset" # 'reset' or 'grasp'

    # Viser GUI Controls
    with viser_server.gui.add_folder("Controller"):
        gui_reset = viser_server.gui.add_button("Reset (d)")
        gui_grasp = viser_server.gui.add_button("Grasp (r)")
        gui_qd_x = viser_server.gui.add_number("qd_o_des X", initial_value=-1.0, step=0.1)
        gui_qd_y = viser_server.gui.add_number("qd_o_des Y", initial_value=0.0, step=0.1)
        gui_qd_z = viser_server.gui.add_number("qd_o_des Z", initial_value=1.0, step=0.1)
    
    @gui_reset.on_click
    def _(_):
        global controller_mode
        controller_mode = "reset"
        sim_thread.reset_state()
        # Erase force visualization lines
        viser_server.scene.add_line_segments(name="/contact_forces", points=np.empty((0, 2, 3)), line_width=2, colors=(0.0, 0.0, 1.0))
        print("\nMode switched to: RESET")

    @gui_grasp.on_click
    def _(_):
        global controller_mode
        controller_mode = "grasp"
        print("\nMode switched to: GRASP")

    # Start Simulation Thread
    sim_thread = MuJoCoSimThread(env_path)
    sim_thread.start()
    print("Simulation thread started.")
    print("Controls: Press 'd' for Default/Reset, 'r' for Grasp, 'q' to Quit.")

    # Setup terminal for non-blocking input
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            if not sim_thread.is_alive():
                print("Simulation thread ended (viewer closed). Exiting...")
                break

            # 0. Handle Input
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == 'd':
                    controller_mode = "reset"
                    sim_thread.reset_state()
                    print("\nMode switched to: RESET")
                elif key == 'r':
                    controller_mode = "grasp"
                    print("\nMode switched to: GRASP")
                elif key == 'q':
                    sim_thread.stop()
                    break

            # 1. Get State from Sim
            q_a, q_ad, q_o, object_radius = sim_thread.get_state()
            
            # 2. Update Visualization
            viser_robot.update_cfg(q_a)
            visualize_spheres({"object_sphere": {"center": q_o, "radius": object_radius, "color": (0.0, 1.0, 0.0)}})

            # Visualize desired object velocity
            qd_o_des_val = np.array([gui_qd_x.value, gui_qd_y.value, gui_qd_z.value])
            viser_server.scene.add_line_segments(
                name="/qd_o_des_arrow",
                points=np.array([[q_o, q_o + 0.2 * qd_o_des_val]]),
                colors=(1.0, 0.0, 1.0),
                line_width=5.0,
            )

            # 3. Kinematics
            link_poses = get_fk_link_poses(q_a, pin_model, pin_data, list_link_names)
            link_jacobians = get_link_jacobians(q_a, pin_model, pin_data, list_link_names)
            link_dJacobians_dq = get_link_dJacobians_dq(q_a, pin_model, pin_data, list_link_names)

            # 4. Contact Geometry
            contact_sphere_vec, link_J_for_spheres, link_centers_for_sphere, contact_sphere_params, dlink_jacobians_cat_dq = compute_contact_spheres(link_poses, link_jacobians, link_dJacobians_dq)
            visualize_spheres(contact_sphere_params)
            
            contact_pos, phi, contact_flag, n_hat, t_hat, J_c, dphi_dq, dJ_c_dq = compute_contact_states(
                contact_sphere_vec, link_centers_for_sphere, link_J_for_spheres, q_o, object_radius, table_height, dlink_jacobians_cat_dq=dlink_jacobians_cat_dq
            )

            u = np.array(default_q)

            if controller_mode == "grasp":
                # 5. Task Planning
                N_spheres = contact_sphere_vec.shape[0]
                contact_sphere_positions = contact_sphere_vec[:, :3]
                contact_sphere_radii = contact_sphere_vec[:, 3]
                
                transformation_matrix = np.eye(6).reshape(1, 6, 6).repeat(N_spheres, axis=0)
                transformation_matrix[:, 3:, :3] = -batch_skew(contact_sphere_positions - link_centers_for_sphere)
                contact_sphere_jacobians = transformation_matrix @ link_J_for_spheres

                N = 5
                tracking_frame_Jacobians = contact_sphere_jacobians
                finger_tip_idx = [0, 1, 2, 3, 4]
                twist_des = np.zeros((N, 6), dtype=np.float64)
                weights_frame_tracking = np.zeros((N, 6), dtype=np.float64)
                for i in finger_tip_idx:
                    twist_des[i, 3:] = 10.0 * (q_o.reshape(1, 3) - contact_sphere_positions[i, :])
                    if contact_flag[i]:
                        weights_frame_tracking[i, 3:] = 0.1
                    else:
                        weights_frame_tracking[i, 3:] = 0.5
                
                qd_o_des = qd_o_des_val
                weights_object_tracking = 1 * np.ones(3, dtype=np.float64)
                
                # 6. QP Control
                trust_region_bound = 0.01
                m = q_a.shape[0]
                n = m + 3
                L = tracking_frame_Jacobians.shape[0]
                q = np.hstack([q_o, q_a])
                u_bar = q_a.copy()

                P_diag, b, J_c_active, c = compute_kinodynamic_matrices(q_o, q_a, u_bar, contact_pos, phi, contact_flag, J_c)
                q_plus_bar, lambda_plus_bar = forward_dynamics(q_o, q_a, J_c_active, P_diag, b, c)

                dbdq = np.zeros((24, 24), dtype=np.float64)
                dbdq[:3, :3] = -np.diag(eps * M_o) / h**2
                dbdu = np.zeros((24, 21), dtype=np.float64)
                dbdu[3:, :] = -np.diag(K_a)

                dq_plus_dq, dlambda_dq, dq_plus_du, dlambda_du = compute_sensitivity(
                    P_diag, J_c_active, lambda_plus_bar, q, q_plus_bar, c, dbdu, dbdq, dphi_dq[contact_flag], dJ_c_dq[contact_flag], kappa, mu
                )

                J_t_flat = tracking_frame_Jacobians.reshape(-1, m)
                J_cal = np.zeros((3 + 6 * L, n), dtype=np.float64)
                J_cal[:3, :3] = np.eye(3)
                J_cal[3:, 3:] = J_t_flat
                w_q = np.hstack([weights_object_tracking, weights_frame_tracking.reshape(-1)])

                A_q = J_cal @ dq_plus_du
                b_q = (J_cal @ (q_plus_bar - q) - np.hstack([qd_o_des, twist_des.reshape(-1)]) * h)

                Q = 0.5 * A_q.T @ np.diag(w_q) @ A_q + 1.0e-8 * np.eye(m)
                Q = sparse.csc_matrix(Q)
                q_vec = A_q.T @ (w_q * b_q)

                A_mat = sparse.csc_matrix(np.eye(m))
                l = np.maximum(-trust_region_bound, pin_model.lowerPositionLimit - u_bar)
                u_bound = np.minimum(trust_region_bound, pin_model.upperPositionLimit - u_bar)

                osqp_prob = osqp.OSQP()
                osqp_prob.setup(P=Q, q=q_vec, A=A_mat, l=l, u=u_bound, verbose=False)
                result = osqp_prob.solve()
                
                if result.info.status_val == osqp.constant("OSQP_SOLVED"):
                    delta_u = result.x
                else:
                    delta_u = np.zeros((m,), dtype=np.float64)

                u = u_bar + delta_u
                lambda_plus = lambda_plus_bar + (dlambda_du @ delta_u).reshape(-1, 3)
                
                # Visualize predicted forces
                contact_forces = (
                    lambda_plus[:, 0:1] * n_hat[contact_flag, 0, :]
                    + lambda_plus[:, 1:2] * t_hat[contact_flag, 0, :]
                    + lambda_plus[:, 2:3] * t_hat[contact_flag, 1, :]
                )
                visualize_contact_force(contact_pos[contact_flag], contact_forces)
            
            # 7. Send Control to Sim
            u[0] = default_q[0] # Keep z height fixed, can't get the z to work too well.
            sim_thread.set_control(u)

    except KeyboardInterrupt:
        print("Stopping simulation...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        sim_thread.stop()
        sim_thread.join()
        print("Done.")