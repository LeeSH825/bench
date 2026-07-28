from __future__ import annotations

import numpy as np


RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi / 180.0
SECONDS_PER_HOUR = 3600.0
MRP_TO_ANGLE_FACTOR = 4.0


def rad_s_to_deg_h(value):
    return np.asarray(value, dtype=np.float64) * RAD_TO_DEG * SECONDS_PER_HOUR


def deg_h_to_rad_s(value):
    return np.asarray(value, dtype=np.float64) * DEG_TO_RAD / SECONDS_PER_HOUR


def rad_to_deg(value):
    return np.asarray(value, dtype=np.float64) * RAD_TO_DEG


def deg_to_rad(value):
    return np.asarray(value, dtype=np.float64) * DEG_TO_RAD


def mrp_delta_to_rad(delta_sigma):
    return np.asarray(delta_sigma, dtype=np.float64) * MRP_TO_ANGLE_FACTOR


def mrp_delta_to_deg(delta_sigma):
    return mrp_delta_to_rad(delta_sigma) * RAD_TO_DEG


def mrp_norm_to_angle_rad(sigma_norm):
    return MRP_TO_ANGLE_FACTOR * np.arctan(np.asarray(sigma_norm, dtype=np.float64))


def mrp_norm_to_angle_deg(sigma_norm):
    return mrp_norm_to_angle_rad(sigma_norm) * RAD_TO_DEG


def mrp_small_angle_relative_error(sigma_norm):
    norm = np.asarray(sigma_norm, dtype=np.float64)
    exact = mrp_norm_to_angle_rad(norm)
    approx = mrp_delta_to_rad(norm)
    out = np.full_like(exact, np.nan, dtype=np.float64)
    mask = np.abs(exact) > np.finfo(np.float64).eps
    out[mask] = np.abs(approx[mask] - exact[mask]) / np.abs(exact[mask])
    out[~mask] = 0.0
    return out


def mrp_covariance_axis_sigma_deg(covariance):
    cov = np.asarray(covariance, dtype=np.float64)
    diag = np.diagonal(cov, axis1=-2, axis2=-1)
    return mrp_delta_to_deg(np.sqrt(np.maximum(diag, 0.0)))


def mrp_covariance_axis_band_deg(covariance, sigma_multiplier=3.0):
    return np.asarray(sigma_multiplier, dtype=np.float64) * mrp_covariance_axis_sigma_deg(covariance)


def covariance_axis_sigma_deg(covariance, covariance_space):
    cov = np.asarray(covariance, dtype=np.float64)
    diag = np.diagonal(cov, axis1=-2, axis2=-1)
    sigma = np.sqrt(np.maximum(diag, 0.0))
    if covariance_space == "mrp":
        return mrp_delta_to_deg(sigma)
    if covariance_space == "rotation_vector_rad":
        return sigma * RAD_TO_DEG
    raise ValueError(f"unknown covariance_space={covariance_space!r}")


def covariance_axis_band_deg(covariance, covariance_space, sigma_multiplier=3.0):
    return np.asarray(sigma_multiplier, dtype=np.float64) * covariance_axis_sigma_deg(covariance, covariance_space)


def attitude_coordinate_to_deg(value, coordinate_space):
    if coordinate_space == "mrp":
        return mrp_delta_to_deg(value)
    if coordinate_space == "rotation_vector_rad":
        return rad_to_deg(value)
    raise ValueError(f"unknown attitude coordinate_space={coordinate_space!r}")
