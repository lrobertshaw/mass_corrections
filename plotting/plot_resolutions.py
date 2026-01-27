import numpy as np
from matplotlib import pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from plotting import scale, title, iqr, cols


def boxplot(jets):
    _, (ax_pt, ax_mass) = plt.subplots(2, 1, figsize=(scale["figure"], scale["figure"]))

    def _make_bp(ax, variable, factor=3):
        raw, corr = jets[f"{variable}_resp"], jets[f"{variable}_resp_corr"]
        resp_mask = abs(raw) < factor
        vp = ax.violinplot([raw[resp_mask], corr[resp_mask]], vert=False, showextrema=False)

        for pc, c in zip(vp['bodies'], cols):
            pc.set_facecolor(c)
            pc.set_alpha(1.0)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.0)

        bp = ax.boxplot([raw, corr],
            vert=False, tick_labels=["Raw", "Corrected"], sym="x", widths=0.6,
            showmeans=True, meanline=True, medianprops={"linestyle": "-"}, meanprops={"linestyle": ":"},
            showfliers=False)
        for median, c in zip(bp["medians"], cols):
            median.set_color("black")
            median.set_linewidth(1.0)
        for mean, c in zip(bp["means"], cols):
            mean.set_color("black")
            mean.set_linewidth(1.0)

        if variable == "mass": xlab = r"$\frac{M^{L1}}{M^{Gen}}$"
        elif variable == "pt": xlab = r"$\frac{p_{T}^{L1}}{p_{T}^{Gen}}$"
        else: raise Exception("Invalid variable for boxplot!")
        ax.set_xlabel(xlab, fontsize=scale["label"])
        return ax
    
    ax_pt, ax_mass = _make_bp(ax_pt, "pt"), _make_bp(ax_mass, "mass")
    for ax in (ax_pt, ax_mass):
        ax.set_xlim([-0.1, 2.2])
        ax.set_xticks(np.linspace(0, 2, 9))
        ax.invert_yaxis()
        hep.cms.label(ax=ax, llabel = title, rlabel = "200 PU (14 TeV)", fontsize=scale["title"])
        ax.grid()
        ax.legend()
        ax.tick_params(axis='both', which='major', labelsize=scale["ticks"])

    plt.tight_layout()
    plt.show()


def histogram(jets, nBins: int = 201, rng: tuple[float, float] = (0.0, 5.0)):
    _, (ax_pt, ax_mass) = plt.subplots(2, 1, figsize=(scale["figure"], 2*(scale["figure"]/1.5)))

    def _make_hist(ax, variable, nBins=201, rng=(0.0, 5.0)):
        raw_median, raw_mean, raw_var, raw_iqr = np.median(jets[f'{variable}_resp']), np.mean(jets[f'{variable}_resp']), np.var(jets[f'{variable}_resp']), iqr(jets[f'{variable}_resp'])
        corr_median, corr_mean, corr_var, corr_iqr = np.median(jets[f'{variable}_resp_corr']), np.mean(jets[f'{variable}_resp_corr']), np.var(jets[f'{variable}_resp_corr']), iqr(jets[f'{variable}_resp_corr'])

        ax.hist(jets[f"{variable}_resp"], bins=nBins, range=rng, histtype='step', label=f"\n$\\bf{{Raw}}$\nMedian = {raw_median:.2f}\nMean = {raw_mean:.2f}\nVariance = {raw_var:.2f}\nIQR = {raw_iqr:.2f}", color="tab:blue", density=True)
        ax.hist(jets[f"{variable}_resp_corr"], bins=201, range=rng, histtype='step', label=f"\n$\\bf{{Corrected}}$\nMedian = {corr_median:.2f}\nMean = {corr_mean:.2f}\nVariance = {corr_var:.2f}\nIQR = {corr_iqr:.2f}", color="tab:orange", density=True)
        # ax.axvline(np.median(jets[f"{variable}_resp"]), linestyle='--', label=f"", color="tab:blue")
        # ax.axvline(np.mean(jets[f"{variable}_resp"]), linestyle=':', label=f"Raw Mean: {np.mean(jets[f'{variable}_resp']):.3f}", color="tab:blue")
        # ax.axvline(np.median(jets[f"{variable}_resp_corr"]), linestyle='--', label=f"", color="tab:orange")
        # ax.axvline(np.mean(jets[f"{variable}_resp_corr"]), linestyle=':', label=f"Corrected Mean: {np.mean(jets[f'{variable}_resp_corr']):.3f}", color="tab:orange")

        if variable == "mass": xlab = r"$\frac{M^{L1}}{M^{Gen}}$"
        elif variable == "pt": xlab = r"$\frac{p_{T}^{L1}}{p_{T}^{Gen}}$"
        else: raise Exception("Invalid variable for histogram!")
        ax.set_xlabel(xlab, fontsize=scale["label"])
        ax.set_ylabel("Density", fontsize=scale["label"])
        return ax

    ax_pt, ax_mass = _make_hist(ax_pt, "pt", nBins=nBins, rng=rng), _make_hist(ax_mass, "mass", nBins=nBins, rng=rng)
    for ax in (ax_pt, ax_mass):            
        hep.cms.label(llabel = title, rlabel = "200 PU (14 TeV)", fontsize=scale["title"], ax=ax)
        ax.set_xlim([-0.1, 2.2])
        ax.set_xticks(np.linspace(0, 2, 9))
        ax.grid()
        ax.legend(fontsize=scale["legend"]*1.25)
        ax.tick_params(axis='both', which='major', labelsize=scale["ticks"])

    plt.tight_layout()
    plt.show()


def response_plot(jets, metric="mean"):

    def _bin_response(var, nBins=101):
        if var == "pt":
            gen = jets["genpt"]
            bins = np.linspace(0, 500, nBins)
        elif var == "mass":
            gen = jets["genmass"]
            bins = np.linspace(0, 180, nBins)
        else:
            raise Exception("Invalid variable!")

        reco_bias, reco_var = [], []
        corr_bias, corr_var = [], []
        for b in range(len(bins) - 1):
            mask = (gen >= bins[b]) & (gen < bins[b+1])
            recoInBin = jets[f"{var}_resp"][mask]
            corrInBin = jets[f"{var}_resp_corr"][mask]

            reco_bias.append(getattr(np, metric)(recoInBin))
            corr_bias.append(getattr(np, metric)(corrInBin))

            if metric == "mean":
                reco_var.append(np.var(recoInBin, ddof=1))
                corr_var.append(np.var(corrInBin, ddof=1))
            elif metric == "median":
                pass
                reco_var.append( iqr(recoInBin) )
                corr_var.append( iqr(corrInBin) )

        return reco_bias, reco_var, corr_bias, corr_var, bins

    def _plot_response(axs, reco_bias, reco_var, corr_bias, corr_var, bins):
        ax_b, ax_v = axs
        xs = 0.5 * (bins[:-1] + bins[1:])

        ax_b.errorbar(xs, reco_bias,# xerr = xs[0], yerr = ,
                        color="tab:blue", capsize=3, label="Raw")
        ax_b.errorbar(xs, corr_bias, color="tab:orange", capsize=3, label="Corrected")
        ax_b.axhline(1.0, linestyle="--", color="black", label="No bias")

        ax_v.errorbar(xs, reco_var, color="tab:blue", capsize=3, label="Raw")
        ax_v.errorbar(xs, corr_var, color="tab:orange", capsize=3, label="Corrected")
        ax_v.axhline(0.0, linestyle="--", color="black", label="No variance")

        ax_b.set_ylim([-0.05, 2.05])
        ax_v.set_ylim([-0.05, 1.05])

        ax_b.set_ylabel(f"${metric.capitalize()}( \\frac{{L1}}{{Gen}} )$", fontsize=scale["label"])
        if metric == "mean":
            ax_v.set_ylabel("$Variance( \\frac{L1}{Gen} )$", fontsize=scale["label"])
        elif metric == "median":
            ax_v.set_ylabel("$IQR( \\frac{L1}{Gen} )$", fontsize=scale["label"])

        for ax in (ax_b, ax_v):
            hep.cms.label(
                ax=ax,
                llabel=title,
                rlabel="200 PU (14 TeV)",
                fontsize=scale["title"],
            )

        return ax_b, ax_v

    # ---- One figure per variable ----
    for var in ("pt", "mass"):

        _, axs = plt.subplots(
            1, 2, figsize=(2*scale["figure"], 1.25*(scale["figure"]/1.5)), sharex=True
            )

        reco_bias, reco_var, corr_bias, corr_var, bins = _bin_response(var)
        ax_b, ax_v = _plot_response(axs, reco_bias, reco_var, corr_bias, corr_var, bins)

        for ax in (ax_b, ax_v):
            xticks = np.linspace(0, bins[-1], 13 if var == "mass" else 11)
            xlab = "$M^{Gen}$ (GeV)" if var == "mass" else "$p_{T}^{Gen}$ (GeV)"

            ax.set_xticks(xticks)
            ax.set_xlim([bins[0] - bins[1], bins[-1] + bins[1]])
            ax.set_xlabel(xlab, fontsize=scale["label"])
            ax.grid()
            ax.tick_params(axis="both", which="major", labelsize=scale["ticks"])
            ax.legend(fontsize=scale["legend"])

        plt.tight_layout()
        plt.show()


def boxplot_bins(jets):
    
    def _get_binned_boxplots(var, factor=3):
        bins = np.linspace(50, 150, 6) if var == "pt" else np.linspace(10, 40, 6)
        responses, corr_responses, counts = [], [], []
        for b in range(len(bins)-1):
            jetsInBinMask = (jets[f"gen{var}"] >= bins[b]) & (jets[f"gen{var}"] < bins[b+1])
            jetsInBin = jets[jetsInBinMask]

            resp = jetsInBin[f"{var}"] / jetsInBin[f"gen{var}"]
            corr_resp = jetsInBin[f"{var}_corr"] / jetsInBin[f"gen{var}"]
            mask = abs(resp) < factor# & (abs(corr_resp) < factor)

            responses.append( resp[mask] )
            corr_responses.append( corr_resp[mask] )
            counts.append( np.sum(mask) )

        return bins, responses, corr_responses, counts

    def _plot_binned_boxplots(ax, bins, responses, corr_responses, counts, violins=True):
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        x = np.arange(len(bin_centers))
        offset = 0.175
        width = 0.3
        means = False
        # scaled_counts = 2 * np.array(counts) / np.max(counts)
        # ax.bar(x=x, height=scaled_counts, alpha=0.1, color="black", label="Bin population")

        if violins:
            v_raw = ax.violinplot(responses, positions=x - offset, widths=width, showextrema=False)
            v_corr = ax.violinplot(corr_responses, positions=x + offset, widths=width, showextrema=False)
            for raw_body, corr_body in zip(v_raw["bodies"], v_corr["bodies"]):
                raw_body.set_facecolor(cols[0])
                raw_body.set_alpha(1.0)
                raw_body.set_edgecolor("black")
                corr_body.set_facecolor(cols[1])
                corr_body.set_alpha(1.0)
                corr_body.set_edgecolor("black")

        medianprops = {"color": "black", "linestyle": "-", "linewidth": 2}
        meanprops = {"color": "black", "linestyle": ":", "linewidth": 2}
        bp_resp = ax.boxplot(responses, positions=x - offset, widths=width*0.6,
                                showfliers=False, showmeans=means, meanline=means, label="Raw",
                                medianprops=medianprops,
                                meanprops=meanprops, patch_artist=not violins)
        bp_resp_corr = ax.boxplot(corr_responses, positions=x + offset, widths=width*0.6,
                                    showfliers=False, showmeans=means, meanline=means, label="Corrected",
                                    medianprops=medianprops,
                                    meanprops=meanprops, patch_artist=not violins)

        if not violins:
            for box, corr_box in zip(bp_resp["boxes"], bp_resp_corr["boxes"]):
                box.set(facecolor=cols[0], alpha=1.0)
                corr_box.set(facecolor=cols[1], alpha=1.0)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="No bias")

        ax.set_ylim([-0.1, 2.1])

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cols[0], label="Raw"),
                            Patch(facecolor=cols[1], label="Corrected")]
        ax.legend(handles=legend_elements, loc="upper right", fancybox=True, frameon=True)

        ax.grid()
        hep.cms.label(llabel=title, rlabel="200 PU (14 TeV)",
                        fontsize=scale["title"], ax=ax)

        return ax

    # _, axs = plt.subplots( 2, 1, figsize=(scale["figure"], 2*(scale["figure"]/1.5)) )
    for v in ["pt", "mass"]:
        _, ax = plt.subplots( 1, 1, figsize=(scale["figure"], scale["figure"]/1.5) )
        bins, responses, corr_responses, counts = _get_binned_boxplots(v)
        ax = _plot_binned_boxplots(ax, bins, responses, corr_responses, counts)
        xlab = "GeV"    # if v == "mass" else "$p_{T}^{Gen}$ (GeV)"
        ylab = r"$\frac{M^{L1}}{M^{Gen}}$" if v == "mass" else r"$\frac{p_{T}^{L1}}{p_{T}^{Gen}}$"
        
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        x = np.arange(len(bin_centers))
        xticklabels = [ f"${bins[i]:.0f} \leq M^{{Gen}} < {bins[i+1]:.0f}$" for i in range(len(bins)-1)] if v == "mass" else [ f"${bins[i]:.0f} \leq p_{{T}}^{{Gen}} < {bins[i+1]:.0f}$" for i in range(len(bins)-1)]
        ax.set_xticks(x, xticklabels)

        ax.set_xlabel(xlab, fontsize=scale["label"])
        ax.set_ylabel(ylab, rotation=0, fontsize=scale["label"]*1.4)
        ax.yaxis.set_label_coords(-0.08, 0.85)
        ax.tick_params(axis='both', which='major', labelsize=scale["ticks"]/1.25)

        plt.tight_layout()
        plt.show()


def split_histogram(jets):
    _, (ax_pt, ax_mass) = plt.subplots(2, 2, figsize=(2*scale["figure"], 2*(scale["figure"]/1.5)))

    params = {
        "range": [0,3],
        "bins": 101,
        "density": True,
        "histtype": "step"
    }

    # pt_bins = np.linspace(0, 500, 6)
    pt_bins = np.linspace(0, 500, 11)
    pt_norm = plt.Normalize(vmin=0, vmax=len(pt_bins)-2)
    pt_cmap = plt.cm.viridis
    for b in range(len(pt_bins)-1):
        inBinMask = (jets[f"genpt"] >= pt_bins[b]) & (jets[f"genpt"] < pt_bins[b+1])
        ax_pt[0].hist(jets[inBinMask].pt_response, label=f"{int(pt_bins[b])} $\leq$ $p_{{T}}^{{Gen}}$ < {int(pt_bins[b+1])}", color=pt_cmap(pt_norm(b)), **params)
        ax_pt[0].set_xlabel(r"$\frac{p^{L1 (Raw)}_{T}}{p_{T}^{Gen}}$", fontsize=scale["label"])
        ax_pt[1].hist(jets[inBinMask].pt_response_corr, label=f"{int(pt_bins[b])} $\leq$ $p_{{T}}^{{Gen}}$ < {int(pt_bins[b+1])}", color=pt_cmap(pt_norm(b)), **params)
        ax_pt[1].set_xlabel(r"$\frac{p^{L1 (Corr)}_{T}}{p_{T}^{Gen}}$", fontsize=scale["label"])


    # mass_bins = np.linspace(0, 180, 7)
    mass_bins = np.linspace(0, 180, 10)
    mass_norm = plt.Normalize(vmin=0, vmax=len(mass_bins)-2)
    mass_cmap = plt.cm.viridis
    for b in range(len(mass_bins)-1):
        inBinMask = (jets[f"genmass"] >= mass_bins[b]) & (jets[f"genmass"] < mass_bins[b+1])
        ax_mass[0].hist(jets[inBinMask].mass_resp, label=f"{int(mass_bins[b])} $\leq$ $M^{{Gen}}$ < {int(mass_bins[b+1])}", color=mass_cmap(mass_norm(b)), **params)
        ax_mass[0].set_xlabel(r"$\frac{M^{L1 (Raw)}}{M^{Gen}}$", fontsize=scale["label"])
        ax_mass[1].hist(jets[inBinMask].mass_resp_corr, label=f"{int(mass_bins[b])} $\leq$ $M^{{Gen}}$ < {int(mass_bins[b+1])}", color=mass_cmap(mass_norm(b)), **params)
        ax_mass[1].set_xlabel(r"$\frac{M^{L1 (Corr)}}{M^{Gen}}$", fontsize=scale["label"])
        
    for axs in (ax_pt, ax_mass):
        for ax in axs:
            ax.axvline(1.0, linestyle="--", color="black", label="No bias")
            ax.set_xlim([-0.05, 2.05])
            ax.set_ylabel("Density", fontsize=scale["label"])
            ax.set_xticks(np.linspace(0,2,9))
            ax.legend(fontsize=scale["legend"], loc="upper right", fancybox=True, frameon=True)
            hep.cms.label(ax=ax, llabel=title, rlabel="200 PU (14 TeV)", fontsize=scale["title"])
            ax.grid()
            ax.tick_params(axis='both', which='major', labelsize=scale["ticks"])

    plt.tight_layout()
    plt.show()


def plot_all(jets):
    boxplot(jets)
    histogram(jets, nBins=201, rng=(0.0, 5.0))
    response_plot(jets, metric="median")
    boxplot_bins(jets)
    split_histogram(jets)