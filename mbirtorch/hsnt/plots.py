import warnings

import numpy as np


def compare_spectra(spectra_groups, ground_truth=None, labels=None, subtitles=None, title=None, x_label=None, y_label=None, x_lim=None, y_lim=None, wavelengths=None, filename=None,
                    font_size=20, legend_font_size=12, line_width=1.5):
    """
    Function to display and save multiple 2D arrays as images.

    Args:
        spectra_groups(list): list of groups of spectra to display
        ground_truth(list,optional): list of ground truth spectra for comparison
        labels(list,optional): labels for different spectra
        subtitles(list,optional): subtitles for different spectrum groups
        title(str,optional): title for the image
        x_label(str,optional): X axis label
        y_label(str,optional): Y axis label
        x_lim(tuple,optional): (x_min, x_max) to set x-axis display range
        y_lim(tuple,optional): (y_min, y_max) to set y-axis display range
        wavelengths(list,optional): list of wavelength values for the spectra
        filename(str,optional): path to save the image
        font_size(int,optional): base font size; raise for slides. Defaults to 20.
        legend_font_size(int,optional): legend font size. Defaults to 12.
        line_width(float,optional): width of the plotted spectra. Defaults to 1.5.
        """
    import matplotlib.pyplot as plt
    num_groups = len(spectra_groups)
    if num_groups == 0:
        raise ValueError("No spectra groups provided for comparison.")

    num_spectra = len(spectra_groups[0])  # Assume all groups have the same number of spectra

    if labels is None:
        labels = ['Spectrum: ' + str(i+1) for i in range(num_spectra)]

    if wavelengths is None:
        wavelengths = np.arange(len(spectra_groups[0][0]))

    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rc('font', size=font_size)
    plt.figure(figsize=(12, 4 * num_groups))
    plt.suptitle(title)

    for group_idx, spectra in enumerate(spectra_groups):
        ax = plt.subplot(num_groups, 1, group_idx + 1)

        group_labels = labels.copy()
        if ground_truth is not None:
            for i, gt_spectrum in enumerate(ground_truth):
                gt_label = "Ground Truth" if i == 0 else None
                ax.plot(wavelengths, gt_spectrum, 'k--', label=gt_label, lw=line_width)

                # Add signal-to-noise ratio annotation
                err = np.linalg.norm(gt_spectrum - spectra[i])
                snr = 20 * np.log10(np.linalg.norm(gt_spectrum) / err)
                group_labels[i] += f" (SNR: {snr:.1f} dB)"

        for i, spectrum in enumerate(spectra):
            ax.plot(wavelengths, spectrum, label=group_labels[i], lw=line_width)

        if subtitles is not None:
            ax.set_title(subtitles[group_idx])

        # Only add x label on final group
        if group_idx == num_groups - 1:
            ax.set_xlabel(x_label)
        else:
            ax.set_xticklabels([])  # Remove x label
        ax.set_ylabel(y_label)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        ax.legend(loc='lower left', fontsize=legend_font_size)

    if filename is not None:
        try:
            plt.savefig(filename)
        except:
            warnings.warn("Can't write to file.")
