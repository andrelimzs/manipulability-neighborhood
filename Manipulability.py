
import numpy as np
import torch

from differentiable_robot_model.robot_model import DifferentiableRobotModel, DifferentiableFrankaPanda

def compute_manipulability_ellipsoid(robot, ee_link, q):
    """Compute manipulability ellipsoid (vector in R6) of given joint configuration

    ME is given by U*S from the SVD { J = U * S * V^T }

    Args
        q (torch.tensor): Joint configuration (angles)

    Returns
        ME (torch.tensor): Manipulability ellipsoid
    """
    # Get robot jacobian
    J_linear, J_angular = robot.compute_endeffector_jacobian(q, ee_link)
    J = torch.concat([J_linear, J_angular], dim=1)

    # SVD
    U,S,Vh = torch.linalg.svd(J)

    # Broadcast multiplication along batch dim(1)
    ME = torch.matmul(U,S.unsqueeze(-1))

    # Return as (batch, 6)
    return ME.squeeze()

def compute_manipulability_index(robot, ee_link, q):
    """Compute manipulability index (scalar value) of given joint configurations

    MI is given by MI = prod(s1, ..., sn)
    
    Args
        q (torch.tensor): Joint configuration (angles)

    Returns
        MI (torch.tensor): Manipulability Index    
    """
    # Get robot jacobian
    J_linear, J_angular = robot.compute_endeffector_jacobian(q, 'panda_virtual_ee_link')
    J = torch.concat([J_linear, J_angular], dim=1)

    # SVD
    U,S,Vh = torch.linalg.svd(J)

    # Manipulability index MI = product(s_i) for all 'i's
    MI = torch.prod(S, dim=1)

    return MI

def get_approximate_neighborhood(q_points, mag=np.deg2rad(0.1)):
    """Compute an (approximate) neighborhood of points about a configuration in joint space

    Approximate by independently perturbing each joint angle, instead of
        computing the cartesian product of all perturbations
    
    Args
        q_points (torch.tensor): Point
        
    Returns
        out (list(torch.tensor)): List of neighbors (each set of neighbors is a tensor)
    """
    out = []
    for q in q_points:
        neighborhood = []
        for joint in range(1,7):
            for delta in [-1, 1]:
                tmp = q.clone()
                # q_points is (N x 7)
                tmp[joint] += delta * mag
                neighborhood.append(tmp)
        out.append(torch.stack(neighborhood, 0))
    return out

def get_neighborhood(q_points, num_neighbors=10, mag=np.deg2rad(0.1)):
    """Compute a neighborhood of points about a configuration in joint space

    The neighborhood is the cartesian product of all joint angle perturbations
    
    Args
        q_points (torch.tensor): Point
        
    Returns
        out (list(torch.tensor)): List of neighbors (each set of neighbors is a tensor)
    """
    # Calculate the 7D perturbations (N^6, 7)
    tmp = [torch.zeros(1)] + [ torch.linspace(-mag, mag, num_neighbors) for i in range(1,7) ]
    perturbations = torch.cartesian_prod(*tmp)

    out = []
    for q in q_points:
        # Broadcast perturbation (N^6, 7) with q (1,7)
        neighborhood = perturbations + q
        out.append(neighborhood)
    
    return out

def compute_manipulability_neighborhood(robot, ee_link, q_points, num_neighbors=10):
    """Compute the manipulability neighborhood (MN) as the weighted sum of nearby MIs
    
    Define the manipulability neighborhood as the (distance) weighted sum of
    manipulability indices about a point.
    
    Args
        q (torch.tensor): Tensor (Num. of joints) of joint configuration
    Returns
        MI (torch.tensor): Tensor (1) manipulability neighborhood
    """
    # Generate neighborhood of points
    q_neighbors = get_neighborhood(q_points, num_neighbors=num_neighbors)
    
    # Compute MN for each point
    MN = []
    for q, neighbors in zip(q_points, q_neighbors):
        # MI Neighborhood is the sum of MIs for the configurations in the neighborhood
        MI = compute_manipulability_index(robot, ee_link, neighbors)

        # Compute MN as the convolution of MI with distance kernel
        distance = torch.linalg.vector_norm( neighbors - q, dim=1 )

        # Weigh by distance
        MN.append(torch.sum(MI * distance))
    
    # Stack list into tensor and return
    MN = torch.stack(MN)
    return MN