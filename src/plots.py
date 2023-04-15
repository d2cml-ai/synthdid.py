import matplotlib, matplotlib.pyplot as plt, numpy as np, pandas as pd

class Plots:
    def plot_outcomes(self, times = None, time_title_cb = int):
        # matplotlib.use('Agg')
        sdid_weights = self.weights
        lambda_wg = sdid_weights["lambda"]
        omega_wg = sdid_weights["omega"]
        table_result = self.att_info
        N0s, T0s = table_result.N0, table_result.T0
        if times is None:
            times = table_result.time

        Y_setups = self.Y_betas
        t_span = np.sort(np.unique(self.data_ref.time))
        # plots = {}
        plots = []
        for i, time in enumerate(times):
            omega_hat = omega_wg[i]
            lambda_hat = lambda_wg[i]
            N0, T0 = N0s[i], T0s[i]
            Y_year = np.array(Y_setups[i])
            Y_t = np.mean(Y_year[N0:, :], axis=0)
            Y_c = Y_year[:N0, :]
            n_features = Y_c.shape[0]
            Y_sdid_traj = np.dot(omega_hat, Y_c)

            values_traj = np.concatenate((Y_sdid_traj, Y_t))

            plot_y_min = values_traj.min()
            plot_y_max = values_traj.max()
            plot_height = plot_y_max - plot_y_min
            base_plot = plot_y_min - plot_height / 5 

            # plot_zero = 

            range_fill = pd.DataFrame(
                {"line": lambda_hat * plot_height / 3 + base_plot, "time": t_span[:T0]}
            )
            trajectory = pd.DataFrame(
                {
                    "time": t_span,
                    'control': Y_sdid_traj,
                    'treatment': Y_t
                }
            )

            fig, ax = plt.subplots()
            ax.plot("time", "control", label="Control", data=trajectory, linestyle="--")
            ax.plot("time", "treatment", label="Treatment", data=trajectory)
            ax.legend()

            if base_plot < 0 and plot_y_max > 0:
                ax.axhline(y=0, color="grey", linestyle="--", lw=.8, alpha=.3)
            ax.fill_between("time", base_plot, "line", data=range_fill, label="", alpha=0.7, color="#0D7F44")
            ax.axvline(x=times[i], label="", color='k', linestyle="--", lw=.8)

            ax.set_xlabel("Time")
            ax.set_title("Adoption: " + str(time_title_cb(time)));

            # plots[f"t_{time}"] = fig
            plots.append(fig)

        self.plot_outcomes = plots
        return self

    def plot_weights(self, unit_filter=None, times=None, time_title_cb=int):
        weights = self.weights
        table_result = self.att_info
        lambda_wg = weights["lambda"]
        omega_wg = weights["omega"]
        N0s = table_result.N0
        T0s = table_result.T0
        T1s = table_result.T1
        N1s = table_result.N1
        atts = np.round(table_result.att_time, 2)
        y_units = self.Y_units
        Y_setups = self.Y_betas

        if times is None:
            times = table_result.time

        real_att = np.round(self.att, 2)

        # label_size 
        ls = np.arange(0, 114, 1)
        ls_rel = np.interp(ls, (ls.min(), ls.max()), (9, 4))#[len(weights_dots) - 1]
        # ns = ls_rel[0]
        if unit_filter is not None: 
            l_unit_f = len(unit_filter)
            if l_unit_f > 114:
                ns = ls_rel[113]
            ns = ls_rel[l_unit_f - 1]
        plots = []
        def plot_times(i):
            
            N0, T0, N1, T1 = N0s[i], T0s[i], N1s[i], T1s[i]

            units = y_units[i][:N0]
            ns = ls_rel[len(units) - 1]
            Y = Y_setups[i].to_numpy()

            lambda_hat = lambda_wg[i]
            omega_hat = omega_wg[i]
            lambda_pre = np.concatenate((lambda_hat, np.full(T1, 0)))
            lambda_post = np.concatenate((np.full(T0, 0), np.full(T1, 1 / T1)))
            omega_control = np.concatenate((omega_hat, np.full(N1, 0)))
            omega_treat = np.concatenate((np.full(N0, 0), np.full(N1, 1 / N1)))

            difs = np.dot(omega_treat, Y).dot(lambda_post - lambda_pre) -\
                np.dot(Y[:N0, :], (lambda_post - lambda_pre))
            size_dot = omega_hat / np.max(omega_hat) * 10
            color_dot = np.where(size_dot == 0, "#9D0924", "#2897E2")
            # shape_dot = np.where(size_dot == 0, ".", "v")
            spaces = " " * (len(str(int(times[i]))) + 1)
            import matplotlib.pyplot as plt

            size_dot = np.interp(size_dot, (size_dot.min(), size_dot.max()), (1, 50))

            weights_dots = pd.DataFrame({
                "unit": units, "difs": difs, "size": size_dot, 
                # "shape": shape_dot, 
                "color": color_dot
                })

            if unit_filter is not None:
                weights_dots = weights_dots.query("unit in @unit_filter")
                size_dot = weights_dots.size
                size_dot = np.interp(size_dot, (size_dot.min(), size_dot.max()), (1, 50))
                weights_dots["size_dot"] = size_dot

            fig, ax = plt.subplots()

            ax.scatter("unit", "difs", data = weights_dots, s = "size", c = "color", label = "")
            # ax.scatter("unit", "difs", data = weights_dots, s = "size", color = "color", marker=shape_dot)
            ax.set_xticklabels(units, rotation = 90, fontsize = ns);
            ax.set_xlabel("Group")
            ax.set_ylabel("Difference")
            ax.set_title("Adoption: " + str(time_title_cb(times[i])))
            ax.axhline(y=atts[i], linestyle = "--", color = "#E00029", lw = .6, label = f"Att {time_title_cb(times[i])}: {atts[i]}")
            ax.axhline(y=self.att, linestyle = "--", color = "#640257", lw = .8, label = f"Att: {spaces} {real_att}")
            ax.legend(fontsize = 9)
            if weights_dots.difs.max() > 0 and weights_dots.difs.min() < 0:
                ax.axhline(y=0, lw = .5, c = "#0D3276")
            plots.append(fig)

        for i, time in enumerate(times):
            plot_times(i)
        self.plot_weights = plots
        return self
