import streamlit as st
import torch
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Reconstruct import load_checkpoint, PINNSystem
from ReconstructORI import load_checkpoint as load_checkpoint_ori
from ReconstructORI import PINNSystem as PINNSystemORI
import glob
import os
from scipy.spatial import Voronoi
from shapely.geometry import Polygon


base_dir = os.path.dirname(os.path.abspath(__file__))


# Page setup
st.set_page_config(layout="wide", page_title="PINN Model Viewer")
st.title("POD-PINN Model Visualization System")


# Calculate Weights for Each Point
def estimate_weights(coordinates):
    """
    Estimate area-based weights for each point using 2D Voronoi diagram.

    Args:
        coordinates (ndarray): N x 2 array of 2D coordinates.

    Returns:
        ndarray: N-length array of area weights for each input point.
    """
    vor = Voronoi(coordinates)
    weights = np.zeros(len(coordinates))

    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if -1 in region or len(region) == 0:
            # Skip infinite or degenerate regions
            weights[i] = 0
            continue
        polygon_points = [vor.vertices[j] for j in region]
        try:
            poly = Polygon(polygon_points)
            area = poly.area
            weights[i] = area
        except:
            weights[i] = 0  # fallback in case polygon is invalid

    # Optional: Normalize weights to sum to 1
    total = np.sum(weights)
    if total > 0:
        weights /= total

    return weights


# Initiation
# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = ""
if 'model_info' not in st.session_state:
    st.session_state.model_info = {}
if 'exp_data' not in st.session_state:
    st.session_state.exp_data = pd.read_csv(os.path.join(base_dir, 'EXP.csv'))
if 'coordinates' not in st.session_state:
    st.session_state.coordinates = pd.read_csv(os.path.join(base_dir, 'coordinate.csv'), header=None).values
    st.session_state.weights = estimate_weights(st.session_state.coordinates)


# Get all available model files
def get_model_files():
    """Get all model files from Reconstruct and ReconstructORI folders"""
    models = []
    # Get models from Reconstruct folder
    for file in glob.glob(os.path.join(base_dir, "Reconstruct/*.pth")):
        models.append(file)
    # Get models from ReconstructORI folder
    for file in glob.glob(os.path.join(base_dir, "ReconstructORI/*.pth")):
        models.append(file)
    return sorted(models)


# Model loading function
def load_model(file_path):
    """Load model by file path and parse parameters"""
    try:
        # Determine model type
        if "ReconstructORI" in file_path:
            model = load_checkpoint_ori(file_path)
            model_type = "ReconstructORI"
        else:
            model = load_checkpoint(file_path)
            model_type = "Reconstruct"

        # Parse filename parameters
        filename = os.path.basename(file_path)
        pattern = r"Checkpoint_SEED(\d+)_LR([\d.]+)_HD(\d+)_HL(\d+)_Epoch(\d+)_WithPhysics(\d)_WithBatch(\d)_Pretrain(\d+)\.pth" \
            if "Pretrain" in file_path else r"Checkpoint_SEED(\d+)_LR([\d.]+)_HD(\d+)_HL(\d+)_Epoch(\d+)_WithPhysics(\d)_WithBatch(\d)\.pth"
        match = re.match(pattern, filename)

        if not match:
            st.error("Filename format does not match!")
            return None, "", {}

        params = {
            "SEED": int(match.group(1)),
            "LR": float(match.group(2)),
            "HD": int(match.group(3)),
            "HL": int(match.group(4)),
            "Epoch": int(match.group(5)),
            "WithPhysics": bool(int(match.group(6))),
            "WithBatch": bool(int(match.group(7))),
            "Pretrain": int(match.group(8)) if "Pretrain" in file_path else 0
        }

        params_to_display = {
            "Random Seed": int(match.group(1)),
            "Learning Rates": float(match.group(2)),
            "Hidden Layer Dimensions": int(match.group(3)),
            "Hidden Layers": int(match.group(4)),
            "Epochs": int(match.group(5)),
            "Using Physics Constraints": bool(int(match.group(6))),
            "Traverse All Spatial Coordinates": bool(int(match.group(7))),
            "Pretrain Epochs": int(match.group(8)) if "Pretrain" in file_path else 0
        }

        # Find corresponding loss curve image
        loss_file = filename.replace("Checkpoint", "Losscurve")
        loss_file = loss_file.replace(".pth", ".png")

        loss_file = loss_file.replace(f"_Epoch{params['Epoch']}",
                                      f"_Epoch{params['Epoch'] + params['Pretrain']}")
        loss_path = os.path.join(os.path.dirname(file_path), loss_file)

        return model, model_type, params_to_display, loss_path

    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, "", {}, ""


# Model loading section
st.sidebar.header("Model Selection")

# Get all available models
model_files = get_model_files()
default_model = "ReconstructORI/Checkpoint_SEED42_LR0.01_HD30_HL2_Epoch63_WithPhysics1_WithBatch1_Pretrain50.pth"
default_model = os.path.join(base_dir, default_model)

# Ensure default model exists
if default_model not in model_files and model_files:
    default_model = model_files[0]
if default_model in model_files and model_files:
    default_model = model_files[0]

# Create file selector
selected_model = st.sidebar.selectbox("Select model file", model_files,
                                      index=model_files.index(default_model) if default_model in model_files else 0)

# Add refresh button
if st.sidebar.button("Refresh model list"):
    model_files = get_model_files()
    st.sidebar.success(f"Found {len(model_files)} model files")

# Load model button
if st.sidebar.button("Load selected model"):
    with st.spinner("Loading model..."):
        model, model_type, params, loss_path = load_model(selected_model)
        if model:
            st.session_state.model = model
            st.session_state.model_type = model_type
            st.session_state.model_info = params
            st.session_state.loss_path = loss_path
            st.sidebar.success(f"Model loaded successfully: {os.path.basename(selected_model)}")

# Show model info
if st.session_state.model:
    st.sidebar.subheader("Model Parameters")
    for k, v in st.session_state.model_info.items():
        st.sidebar.text(f"{k}: {v}")

    # Show loss curve
    if os.path.exists(st.session_state.loss_path):
        st.sidebar.image(st.session_state.loss_path, caption="Training Loss Curve")
    else:
        st.sidebar.warning("Loss curve image not found!")

# Input parameter section
st.header("Operating Condition Settings")
col1, col2 = st.columns(2)


# Initialize session_state variables if not exist
if 'omega' not in st.session_state:
    st.session_state['omega'] = 500
if 'qv' not in st.session_state:
    st.session_state['qv'] = 0.25


# Callback functions to sync inputs
def update_omega_slider():
    st.session_state['omega'] = st.session_state['omega_slider']


def update_omega_input():
    st.session_state['omega'] = st.session_state['omega_input']


def update_qv_slider():
    st.session_state['qv'] = st.session_state['qv_slider']


def update_qv_input():
    st.session_state['qv'] = st.session_state['qv_input']


with col1:
    omega_slider = st.slider(
        "Rotating Speed (rpm)",
        375, 675,
        st.session_state['omega'],
        1,
        key="omega_slider",
        on_change=update_omega_slider
    )
    omega_input = st.number_input(
        "Rotating Speed (rpm)",
        375, 675,
        st.session_state['omega'],
        1,
        key="omega_input",
        on_change=update_omega_input
    )

with col2:
    qv_slider = st.slider(
        "Volumetric Flow Rate (m³/s)",
        0.05, 0.48,
        st.session_state['qv'],
        0.01,
        key="qv_slider",
        on_change=update_qv_slider
    )
    qv_input = st.number_input(
        "Volumetric Flow Rate (m³/s)",
        0.05, 0.48,
        st.session_state['qv'],
        0.01,
        key="qv_input",
        on_change=update_qv_input
    )
# Use st.session_state['omega'] and st.session_state['qv'] as the synced values
omega = st.session_state['omega']
qv = st.session_state['qv']


# Model prediction function
def predict_field(coords, omega, qv):
    """Predict flow field using model"""
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    condition = torch.tensor([omega, qv], dtype=torch.float32)
    with torch.no_grad():
        predictions = st.session_state.model(coords_tensor, condition)
    return predictions


# Field plotting function
def plot_field(coords, field, title, vmin=None, vmax=None):
    """Plot flow field distribution"""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Use radial and axial coordinates
    theta = coords[:, 1]
    z = coords[:, 2]

    sc = ax.scatter(theta, z, c=field, cmap='viridis', s=5, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, label=title.split(' ')[0])

    ax.set_xlabel('Theta Coordinate (rad)')
    ax.set_ylabel('Axial Coordinate (m)')
    ax.set_title(title)
    fig.tight_layout()
    return fig


# Error Statistical Function (Hist&CDF)

def plot_error(rel_errors, name, limit=90):
    sorted_errors = np.sort(rel_errors)
    cdf = np.arange(len(sorted_errors)) / len(sorted_errors)

    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Left: Histgram
    color_hist = '#FFA853'
    counts, bins, patches = ax1.hist(
        rel_errors,
        bins=1000,
        density=True,  # Normalization
        color=color_hist,
        alpha=0.6,
        edgecolor='black'
    )
    bin_width = bins[1] - bins[0]
    percent_counts = counts * bin_width * 100
    for rect, pc in zip(patches, percent_counts):
        rect.set_height(pc)
    # Only Show 90%
    ax1.set_xlim([0, np.percentile(sorted_errors, limit)])
    ax1.set_xlabel(f"Relative Error of {name}")
    ax1.set_ylabel("Percentage (%)", color=color_hist)
    ax1.set_ylim([0, max(percent_counts) * 1.1])
    ax1.tick_params(axis='y', labelcolor=color_hist)

    # Right: CDF
    ax2 = ax1.twinx()
    color_cdf = '#92B8F9'
    ax2.plot(sorted_errors, cdf, color=color_cdf, linestyle='-', marker='o', linewidth=2, label='CDF', alpha=0.2)
    ax2.set_ylabel("Cumulative Probability", color=color_cdf)
    ax2.tick_params(axis='y', labelcolor=color_cdf)

    # Title
    fig.suptitle(f"Relative Error Distribution and CDF for {name}", fontsize=12)
    fig.tight_layout()
    return fig


LIMIT = st.slider(
    label="Error Display Cut-off Percentile",
    min_value=50,
    max_value=100,
    value=90,
    step=1,
    help="Only show error distribution for the lowest X% relative errors"
)

# Run prediction when button clicked
if st.button("Compute Flow Field") and st.session_state.model:
    try:
        # Convert to tensor, condition = torch.tensor([630, 0.65])
        condition = torch.tensor([omega, qv], dtype=torch.float32)

        # Predict flow field
        with st.spinner("Computing..."):
            predictions = predict_field(st.session_state.coordinates, omega, qv)

        # Separate physical quantities
        P_pred = predictions['P'].cpu().detach().numpy()
        Ur_pred = predictions['Ur'].cpu().detach().numpy()
        Ut_pred = predictions['Ut'].cpu().detach().numpy()
        Uz_pred = predictions['Uz'].cpu().detach().numpy()

        # Show prediction results
        st.subheader("Model Prediction Results")
        cols = st.columns(4)
        with cols[0]:
            st.pyplot(plot_field(st.session_state.coordinates, P_pred, "Pressure Field (Pa)"))
        with cols[1]:
            st.pyplot(plot_field(st.session_state.coordinates, Ur_pred, "Radial Velocity (m/s)"))
        with cols[2]:
            st.pyplot(plot_field(st.session_state.coordinates, Ut_pred, "Tangential Velocity (m/s)"))
        with cols[3]:
            st.pyplot(plot_field(st.session_state.coordinates, Uz_pred, "Axial Velocity (m/s)"))

        # Check experimental data
        exp_match = st.session_state.exp_data[
            (st.session_state.exp_data['omega'] == omega) & (st.session_state.exp_data['qv'] == qv)
            ]

        if not exp_match.empty:
            exp_idx = exp_match.index[0]
            dataset_type = "Test Set" if exp_match.iloc[0]['test'] else "Training Set"

            # Load experimental data
            P_exp = pd.read_csv(os.path.join(base_dir, 'P.csv'), header=None).iloc[:, exp_idx].values
            Ur_exp = pd.read_csv(os.path.join(base_dir, 'Ur.csv'), header=None).iloc[:, exp_idx].values
            Ut_exp = pd.read_csv(os.path.join(base_dir, 'Ut.csv'), header=None).iloc[:, exp_idx].values
            Uz_exp = pd.read_csv(os.path.join(base_dir, 'Uz.csv'), header=None).iloc[:, exp_idx].values

            # Calculate and show Point-to-Point Relative Errors
            st.subheader("Point-to-Point Error Distribution (POD-PINN comparing to CFD)")
            new_cols = st.columns(4)

            # calculate relative errors
            def relative_errors(pred, true, with_mask=False):
                # when velocity is close to 0, will cause unreasonable result
                # relative error is not suitable when closing to zero
                # the level of velocity is 1e1, thus we use 0.2 as the threshold according to the 1/50 standard
                if with_mask:
                    mask = np.abs(true) > 0.2
                    result = np.abs(pred[mask] - true[mask]) / np.abs(true[mask])
                else:
                    result = np.abs(pred - true) / np.abs(true)
                return result

            with new_cols[0]:
                st.pyplot(plot_field(st.session_state.coordinates, relative_errors(P_pred, P_exp), "Pressure-RelativeError", vmin=0, vmax=2))
            with new_cols[1]:
                st.pyplot(plot_field(st.session_state.coordinates, relative_errors(Ur_pred, Ur_exp), "RadialVelocity-RelativeError", vmin=0, vmax=2))
            with new_cols[2]:
                st.pyplot(plot_field(st.session_state.coordinates, relative_errors(Ut_pred, Ut_exp), "TangentialVelocity-RelativeError", vmin=0, vmax=2))
            with new_cols[3]:
                st.pyplot(plot_field(st.session_state.coordinates, relative_errors(Uz_pred, Uz_exp), "AxialVelocity-RelativeError", vmin=0, vmax=2))

            # Calculate and show the Cumulative Distribution of Relative Error
            st.subheader("Point-to-Point Error Statistics")
            new_cols1 = st.columns(4)

            rel_error_p = relative_errors(P_pred, P_exp, with_mask=True)
            rel_error_ur = relative_errors(Ur_pred, Ur_exp, with_mask=True)
            rel_error_ut = relative_errors(Ut_pred, Ut_exp, with_mask=True)
            rel_error_uz = relative_errors(Uz_pred, Uz_exp, with_mask=True)

            with new_cols1[0]:
                st.pyplot(plot_error(rel_error_p, "P", LIMIT))
            with new_cols1[1]:
                st.pyplot(plot_error(rel_error_ur, "Ur", LIMIT))
            with new_cols1[2]:
                st.pyplot(plot_error(rel_error_ut, "Ut", LIMIT))
            with new_cols1[3]:
                st.pyplot(plot_error(rel_error_uz, "Uz", LIMIT))

            # Compute Correlation Coefficient-Pearson
            def correlation_coefficient(y_pred, y_true):
                mean_pred, mean_true = np.mean(y_pred), np.mean(y_true)
                # due to the result
                return np.sum((y_pred-mean_pred)*(y_true-mean_true))/np.linalg.norm(y_true-mean_true)/np.linalg.norm(y_pred-mean_pred)

            # Compute Area-Weighted Relative Error, needs to be normalized respectively to-
            # -exclude the noise causing by the CFD data extraction method
            def weighted_relative_error(y_pred, y_true):
                """
                Compute the area-weighted relative error between predicted and true values,
                normalized using only the ground truth (y_true) min and max.

                Args:
                    y_pred (np.ndarray): predicted values
                    y_true (np.ndarray): ground truth values

                Returns:
                    float: weighted relative error
                """
                # Avoid division by zero by masking small true values
                mask = y_true > 0.01
                # Normalize based on y_true only
                y_true_min, y_true_max = np.min(y_true), np.max(y_true)
                def normalize(data):
                    return (data - y_true_min) / (y_true_max - y_true_min + 1e-8)
                y_true_norm = normalize(y_true)[mask]
                y_pred_norm = normalize(y_pred)[mask]
                # Relative error
                relative_errors = np.abs((y_pred_norm - y_true_norm) / y_true_norm)
                # Area-weighted average
                weights = st.session_state.weights[mask]
                return np.sum(relative_errors * weights) / np.sum(weights)

            coeff_P = correlation_coefficient(P_pred, P_exp)
            coeff_Ur = correlation_coefficient(Ur_pred, Ur_exp)
            coeff_Ut = correlation_coefficient(Ut_pred, Ut_exp)
            coeff_Uz = correlation_coefficient(Uz_pred, Uz_exp)
            wr_P = weighted_relative_error(P_pred, P_exp)
            wr_Ur = weighted_relative_error(Ur_pred, Ur_exp)
            wr_Ut = weighted_relative_error(Ut_pred, Ut_exp)
            wr_Uz = weighted_relative_error(Uz_pred, Uz_exp)

            metrics_df = pd.DataFrame({
                "Metric": ["Pressure", "Radial Velocity (~0, Not important in an Axial Pump)", "Tangential Velocity", "Axial Velocity"],
                "Correlation Coefficient": [f"{coeff_P:.3%}", f"{coeff_Ur:.3%}", f"{coeff_Ut:.3%}", f"{coeff_Uz:.3%}"],
                "Relative Error (Weighted According to Area)": [f"{wr_P:.3%}", f"{wr_Ur:.3%}", f"{wr_Ut:.3%}", f"{wr_Uz:.3%}"]
            })

            st.subheader("Overall Error Analysis")
            st.write(f"Data Source: {dataset_type}")
            st.dataframe(metrics_df)

        else:
            st.warning("No matching experimental data found for the selected parameters.")

        if st.button("Save Result"):
            # Save result to CSV
            save_path = f"/result/pred_{omega}_{qv}.csv"
            os.makedirs(os.path.join(base_dir, save_path), exist_ok=True)
            df_save = pd.DataFrame({
                'P_pred': P_pred,
                'Ur_pred': Ur_pred,
                'Ut_pred': Ut_pred,
                'Uz_pred': Uz_pred,
            })
            df_save.to_csv(save_path, index=False)
            st.success(f"Results saved to {save_path}")

    except Exception as e:
        st.error(f"Error during computation: {str(e)}")
else:
    if not st.session_state.model:
        st.info("Please load a model first.")
