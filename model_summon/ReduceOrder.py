import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def pod_decomposition(input_file, modes_output_file, coeff_output_file, eigvals_output_file, mean_output_file, figure_output_file, k=None):
    # 1. Load data
    data = np.loadtxt(input_file, delimiter=',').T  # Assume data is a CSV file with comma delimiter
    print("Data loaded. Shape:", data.shape)

    # 2. Calculate the mean of the data and center the data
    data_mean = np.mean(data, axis=1)  # Calculate the mean for each row
    print("Data_Mean_Shape: ", data_mean.shape)
    data_centered = data - data_mean[:, np.newaxis]  # Center the data by subtracting the mean

    # 3. Compute the covariance matrix
    cov_matrix = np.cov(data_centered, rowvar=False)

    # 4. Perform eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # 5. Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[sorted_indices]
    eigvecs_sorted = eigvecs[:, sorted_indices]

    # 6. Select the first k modes (if k is not specified, select all modes)
    if k is None:
        k = data.shape[1]  # Select all modes

    selected_eigvecs = eigvecs_sorted[:, :k]
    selected_eigvals = eigvals_sorted[:k]

    # 7. Calculate modal coefficients (projection coefficients)
    modal_coefficients = np.dot(data_centered, selected_eigvecs)

    # 8. Save mean values to CSV file
    mean_df = pd.DataFrame(data_mean, columns=['Mean'])
    mean_df.to_csv(mean_output_file, index=False)
    print(f"Mean values saved to {mean_output_file}")

    # 9. Save modes to CSV file
    modal_df = pd.DataFrame(selected_eigvecs, columns=[f'Mode {i + 1}' for i in range(k)])
    modal_df.to_csv(modes_output_file, index=False)
    print(f"Modes saved to {modes_output_file}")

    # 10. Save modal coefficients for each observation to CSV file
    coeff_df = pd.DataFrame(modal_coefficients, columns=[f'Coeff Mode {i + 1}' for i in range(k)])
    coeff_df.to_csv(coeff_output_file, index=False)
    print(f"Modal coefficients saved to {coeff_output_file}")

    # 11. Save eigenvalues to CSV file
    eigvals_df = pd.DataFrame(selected_eigvals, columns=['Eigenvalue'])
    eigvals_df.to_csv(eigvals_output_file, index=False)
    print(f"Eigenvalues saved to {eigvals_output_file}")

    # 12. Calculate the energy contribution of each mode (eigenvalue / total eigenvalues sum)
    total_energy = np.sum(selected_eigvals)
    energy_percentages = (selected_eigvals / total_energy) * 100

    # 13. Plot the energy contribution bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, k + 1), energy_percentages, color='blue', alpha=0.7)
    plt.xlabel('Mode Number')
    plt.ylabel('Energy Contribution (%)')
    plt.title('Energy Contribution of Each Mode')
    plt.xticks(range(1, k + 1))
    plt.savefig(figure_output_file)

    # Return modal coefficients, eigenvalues, eigenvectors, energy percentages, mean and original data
    return modal_coefficients, selected_eigvals, selected_eigvecs, energy_percentages, data_mean, data


def reconstruct_data(modal_coefficients, eigvecs, data_mean):
    # Reconstruct data using modal coefficients and eigenvectors
    reconstructed_data = np.dot(modal_coefficients, eigvecs.T)  # Reconstruct data
    # Add the mean back to the reconstructed data to restore original scale
    return reconstructed_data + data_mean[:, np.newaxis]


def evaluate_reconstruction(original_data, reconstructed_data):
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(original_data, reconstructed_data)
    # Calculate correlation coefficient matrix
    correlation_matrix = np.corrcoef(original_data.T, reconstructed_data.T)
    correlation_coefficient = correlation_matrix[:original_data.shape[1], original_data.shape[1]:]
    return mse, correlation_coefficient


if __name__ == '__main__':
    # Run POD decomposition
    outputs = ["P", "Ur", "Ut", "Uz"]
    outdirs = ["ReducedResults"]
    for outdir in outdirs:
        for output in outputs:
            input_file = f'./{output}.csv'  # Input file path
            modes_output_file = f'./{outdir}/modes_{output}.csv'  # Output file path for modes
            coeff_output_file = f'./{outdir}/coefficients_{output}.csv'  # Output file path for coefficients
            eigvals_output_file = f'./{outdir}/eigvals_{output}.csv'  # Output file path for eigenvalues
            mean_output_file = f'./{outdir}/mean_{output}.csv'  # Output file path for mean values
            figure_output_file = f'./{outdir}/EnergySort_{output}.png'  # Output file path for energy bar chart
            reconstructed_output_file = f'./{outdir}/reconstructed_{output}.csv'  # Output file path for reconstructed data

            # Perform POD decomposition
            modal_coefficients, eigvals, eigvecs, energy_percentages, data_mean, original_data = pod_decomposition(
                input_file, modes_output_file, coeff_output_file, eigvals_output_file, mean_output_file,
                figure_output_file, k=8)

            # Print eigenvalues and energy percentages
            print("Eigenvalues:", eigvals)
            print("Energy percentages:", energy_percentages)

            # Reconstruct data using saved modal coefficients and eigenvectors
            reconstructed_data = reconstruct_data(modal_coefficients, eigvecs, data_mean)
            print("Reconstructed data shape:", reconstructed_data.shape)

            # Save reconstructed data
            reconstructed_df = pd.DataFrame(reconstructed_data)
            reconstructed_df.to_csv(reconstructed_output_file, index=False)
            print(f"Reconstructed data saved to {reconstructed_output_file}")

            # Evaluate reconstruction performance
            mse, correlation_coefficient = evaluate_reconstruction(original_data, reconstructed_data)
            print(f"Mean Squared Error (MSE): {mse}")
            print(f"Correlation Coefficient between original and reconstructed data:\n{correlation_coefficient}")
