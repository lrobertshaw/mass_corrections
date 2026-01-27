import numpy as np
from matplotlib import pyplot as plt
import mplhep as hep
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import LogLocator, ScalarFormatter
from plotting import scale, title
hep.style.use("CMS")

class HistoScaleFactors:
    def __init__(self, edges, counts, **scale_factors):
        self.edges = edges
        self.counts = counts
        self.scale_factors = scale_factors

        if len(self.edges) == 2:
            self.plot_bin_densities(self.edges, self.counts)
            self.plot_scale_factors_2d(self.edges, self.scale_factors["pt"], var="pt")
            self.plot_scale_factors_2d(self.edges, self.scale_factors["mass"], var="mass")
            self.plot_scale_factors_1d(self.edges, self.scale_factors["pt"], self.scale_factors["mass"])
        if len(self.edges) == 3:
            # Assuming scale_factors is a tuple: (pt_sfs, mass_sfs, edges, counts)
            self.scale_factors_heatmap(self.edges, "barrel", **self.scale_factors)
            self.plot_bin_densities_3d(self.edges, self.counts, "barrel")

    @staticmethod
    def plot_bin_densities(edges, counts):
        plt.figure(figsize=(scale["figure"], scale["figure"]/1.5))
        pt_edges, eta_edges = edges

        # ylim=204    # cap y so the colours make sense
        # cap_idx = np.argwhere(pt_edges > ylim)[0][0]
        # counts = counts[:cap_idx]
        # pt_edges = pt_edges[:cap_idx+1]

        counts = counts / np.max(counts)

        im = plt.pcolormesh(eta_edges, pt_edges, counts, shading="auto")
        cbar = plt.colorbar(im, label="Counts")
        cbar.ax.tick_params(labelsize=scale["ticks"])
        cbar.set_label(label = "Relative counts", size=scale["label"])
        plt.yscale("log")
        plt.gca().yaxis.set_major_locator(LogLocator(base=10))
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        
        plt.yticks([15, 30, 60, 120, 240, 480, 960, 1920], fontsize=scale["ticks"])
        plt.ylim([10, pt_edges[-1]])

        plt.xlim([-2.5, 2.5])
        plt.xticks(np.linspace(-2.5, 2.5, 11), fontsize=scale["ticks"])

        plt.axvline(-1.5, linestyle="--", color="black", label="Barrel-Endcap border")
        plt.axvline(1.5, linestyle="--", color="black")

        plt.xlabel("$\eta$ bin edges", fontsize=scale["label"])
        plt.ylabel("$p_T^{L1}$ bin edges (GeV)", fontsize=scale["label"])
        plt.minorticks_off()
        hep.cms.label(llabel=title, rlabel="200 PU (14 TeV)", fontsize=scale["title"]*0.8)
        # plt.savefig(format="pdf")
        plt.show()

    @staticmethod
    def plot_scale_factors_2d(edges, sfs, var=None):
        plt.figure(figsize=(scale["figure"], scale["figure"]/1.5))
        pt_edges, eta_edges = edges
        if var == "pt": var = "Jet $p_T$"
        elif var == "mass": var = "Jet mass"

        # ylim=2045.9    # cap y so the colours make sense
        # cap_idx = np.argwhere(pt_edges > ylim)[0][0]
        # sfs = sfs[:cap_idx]
        # pt_edges = pt_edges[:cap_idx+1]

        my_cmap = plt.get_cmap('coolwarm').copy()
        my_cmap.set_bad(color='white')
        im = plt.pcolormesh(eta_edges, pt_edges, sfs,
                            norm=TwoSlopeNorm(vcenter=1, vmin=0, vmax=4),
                            cmap=my_cmap, shading="auto")
        cbar = plt.colorbar(im)
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.75, 2.5, 3.25, 4.0])
        cbar.ax.tick_params(labelsize=scale["ticks"])
        cbar.set_label(label = f"{var} scale factor", size=scale["label"])
        cbar.ax.minorticks_off()
        plt.yscale("log")
        plt.gca().yaxis.set_major_locator(LogLocator(base=10))
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        
        plt.yticks([15, 30, 60, 120, 240, 480, 960, 1920], fontsize=scale["ticks"])
        # plt.yticks([10, 20, 40, 80, 160, 320, 640, 1280], fontsize=scale["ticks"])
        plt.ylim([10, pt_edges[-1]])

        plt.xlim([-2.5, 2.5])
        plt.xticks(np.linspace(-2.5, 2.5, 11), fontsize=scale["ticks"])

        plt.axvline(-1.5, linestyle="--", color="black", label="Barrel-Endcap border")
        plt.axvline(1.5, linestyle="--", color="black")

        plt.xlabel("$\eta$ bin edges", fontsize=scale["label"])
        plt.ylabel("$p_T^{L1}$ bin edges (GeV)", fontsize=scale["label"])
        plt.minorticks_off()
        hep.cms.label(llabel=title, rlabel="200 PU (14 TeV)", fontsize=scale["title"]*0.8)
        plt.show()

    @staticmethod
    def plot_scale_factors_1d(edges, pt_sfs, mass_sfs, region="barrel"):
        pt_edges, eta_edges = edges

        # η bin centres
        eta_centres = eta_edges[:-1] + 0.5 * np.diff(eta_edges)

        # Region selection
        if region == "barrel":
            eta_mask = (eta_centres >= -1.5) & (eta_centres <= 1.5)
        elif region == "endcap":
            eta_mask = (eta_centres < -2.5) | (eta_centres > 2.5)
        else:
            raise ValueError("Invalid region specified (use 'barrel' or 'endcap')")

        # pT bin centres (linear)
        pt_low  = pt_edges[:-1]
        pt_high = pt_edges[1:]
        xs = 0.5 * (pt_low + pt_high)

        # Asymmetric x errors (required for log scale)
        xerr_low  = xs - pt_low
        xerr_high = pt_high - xs
        xerr = np.vstack([xerr_low, xerr_high])

        # Average scale factors over η region
        ys1 = np.mean(pt_sfs[:, eta_mask], axis=1)
        ys2 = np.mean(mass_sfs[:, eta_mask], axis=1)

        plt.figure(figsize=(scale["figure"], scale["figure"]/1.5))
        plt.errorbar(xs, ys1, xerr=xerr, fmt=" ", capsize=0, label="Jet $p_T$ scale factor", color="tab:blue")
        plt.errorbar(xs, ys2, xerr=xerr, fmt=" ", capsize=0, label="Jet mass scale factor", color="tab:orange")

        # --- log pT axis (copied logic from 2D plot) ---
        plt.xscale("log")
        plt.gca().xaxis.set_major_locator(LogLocator(base=10))
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())

        plt.xticks(
            [15, 30, 60, 120, 240, 480, 960, 1920],
            fontsize=scale["ticks"]
        )
        plt.yticks(fontsize=scale["ticks"])
        
        plt.xlim([9, 2500])
        plt.xlabel(r"$p_T^{L1}$ bin edges (GeV)", fontsize=scale["label"])
        plt.ylabel(f"Mean scale factor ({region})", fontsize=scale["label"])
        hep.cms.label(llabel=title, rlabel="200 PU (14 TeV)", fontsize=scale["title"]*1.0)
        plt.grid(True, which="major")
        plt.legend(fontsize=scale["legend"])
        plt.show()

    @staticmethod
    def scale_factors_heatmap(edges, region: str = "barrel", **scale_factors):
        """
        Function assumes three variables in edges: pt, eta, mass
        """
        # _, axs = plt.subplots(len(scale_factors), 1, figsize=(scale["figure"], scale["figure"]*1.25))
        pt_edges, mass_edges, eta_edges = edges
        eta_centers  = 0.5 * (eta_edges[1:]  + eta_edges[:-1])
        if region == "barrel": eta_mask = (eta_centers >= -1.5) & (eta_centers <= 1.5)
        elif region == "endcap": eta_mask = ( (eta_centers < -2.5) | (eta_centers > 2.5) )
        else: raise Exception("Invalid region specified for heatmap!")

        def _make_hm(ax, sf, var=None):
            heat_mean = np.mean(sf[:, :, eta_mask], axis=2)  # shape (pt_bins, mass_bins)

            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.xaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_major_locator(LogLocator(base=10))

            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(style='plain')

            # plt.yticks([10, 20, 40, 80, 160, 320, 640, 1280], fontsize=scale["ticks"])
            ax.set_xlim([10, pt_edges[-1]])
            ax.set_ylim([1, mass_edges[-1]])

            ax.set_xticks([15, 30, 60, 120, 240, 480, 960, 1920])
            ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128])
            # set axis tick font size
            ax.tick_params(axis='x', labelsize=scale["ticks"])
            ax.tick_params(axis='y', labelsize=scale["ticks"])

            # Labels
            ax.set_xlabel("$p_{T}^{L1}$ bin edges (GeV)", fontsize=scale["label"])
            ax.set_ylabel("$M^{L1}$ bin edges (GeV)", fontsize=scale["label"])

            my_cmap = plt.get_cmap('coolwarm').copy()
            my_cmap.set_bad(color='white')
            hm = ax.pcolormesh(pt_edges, mass_edges, heat_mean.T,
                norm=TwoSlopeNorm(vcenter=1, vmin=0.25, vmax=4.0),
                cmap=my_cmap, shading="auto"
            )
            if var == "pt": lab = f"$p_{{T}}^{{L1}}$ scale factor ({region})"
            elif var == "mass": lab = f"$M^{{L1}}$ scale factor ({region})"
            else: lab = "Scale factor"
            cbar = plt.colorbar(hm, label=lab, ax=ax)
            # cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.75, 2.5, 3.25, 4.0])
            cbar.set_ticks([0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0])
            cbar.ax.tick_params(labelsize=scale["ticks"])
            cbar.ax.minorticks_off()
            ax.tick_params(axis='both', which='minor', length=0)
            return ax

        for var, sf in scale_factors.items():
            _, ax = plt.subplots(1, 1, figsize=(scale["figure"], scale["figure"]/1.5))
            ax = _make_hm(ax, sf, var)
            hep.cms.label(llabel = title, rlabel = "200 PU (14 TeV)", fontsize=scale["title"]*0.9, ax=ax)
            ax.grid()
        
            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_bin_densities_3d(edges, counts, region):
        pt_edges, mass_edges, eta_edges = edges
        eta_centers  = 0.5 * (eta_edges[1:]  + eta_edges[:-1])
        if region == "barrel": eta_mask = (eta_centers >= -1.5) & (eta_centers <= 1.5)
        elif region == "endcap": eta_mask = ( (eta_centers < -2.5) | (eta_centers > 2.5) )
        else: raise Exception("Invalid region specified for heatmap!")

        heat_mean = np.sum(counts[:, :, eta_mask], axis=2)  # shape (pt_bins, mass_bins)
        heat_mean /= np.max(heat_mean)
        plt.figure(figsize=(scale["figure"], scale["figure"]/1.5))

        im = plt.pcolormesh(pt_edges, mass_edges, heat_mean, shading="auto")
        cbar = plt.colorbar(im, label="Counts")
        cbar.ax.tick_params(labelsize=scale["ticks"])
        cbar.set_label(label = "Relative counts (barrel)", size=scale["label"])
        
        plt.yscale("log")
        plt.xscale("log")
        plt.gca().yaxis.set_major_locator(LogLocator(base=10))
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        plt.gca().xaxis.set_major_locator(LogLocator(base=10))
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.xlim([10, pt_edges[-1]])
        plt.ylim([1, mass_edges[-1]])
        plt.xticks([15, 30, 60, 120, 240, 480, 960, 1920])
        plt.yticks([1, 2, 4, 8, 16, 32, 64, 128])
        plt.xlabel("$p_{T}^{L1}$ bin edges (GeV)", fontsize=scale["label"])
        plt.ylabel("$M^{L1}$ bin edges (GeV)", fontsize=scale["label"])


class BDTScaleFactors:
    def __init__(self, model):
        self.plot_feature_importance(model)

    @staticmethod
    def plot_feature_importance(model):
        importance = model.get_score(importance_type='gain')
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        print(importance)

        plt.figure(figsize=(scale["figure"], scale["figure"]/1.5))
        plt.bar(range(len(importance)), list(importance.values()), align='center')
        plt.xticks(range(len(importance)), list(importance.keys()), rotation=90, fontsize=scale["ticks"])
        plt.yticks(fontsize=scale["ticks"])
        plt.xlabel("Features", fontsize=scale["label"])
        plt.ylabel("Importance (gain)", fontsize=scale["label"])
        hep.cms.label(llabel=title, rlabel="200 PU (14 TeV)", fontsize=scale["title"]*0.8)

        plt.show()

    @staticmethod
    def plot_scale_factors(model):
        xs = np.linspace(0, 2048, 2049)
        ys = np.linspace(0, 182, 183)
        np.meshgrid(xs, ys)
        pass


class MLPScaleFactors:
    def __init__(self):
        pass