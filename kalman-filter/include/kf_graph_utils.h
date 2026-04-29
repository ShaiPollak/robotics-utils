#ifndef KF_GRAPH_UTILS_H
#define KF_GRAPH_UTILS_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "kalman_filter.h"   // ESEKF, chi2Threshold

/*
 * Visualization utilities for the ES-EKF.
 *
 * FilterRecorder collects snapshots of the running filter and provides three
 * output paths:
 *
 *   toCSV(path)              Write a flat CSV readable by Python / Excel / MATLAB.
 *   toMatlab(path)           Write a self-contained MATLAB / Octave .m script that
 *                            generates all subplots with ±2σ bands.
 *   showPlot()               Render a static gnuplot window (blocks until closed).
 *   updateLivePlot()         Incrementally refresh a live gnuplot window each cycle.
 *
 * Plots produced (2×2 base layout, extends to 3×2 when ground truth is available):
 *   • Position x/y/z   with ±2σ covariance envelope
 *   • Velocity vx/vy/vz with ±2σ covariance envelope
 *   • Orientation roll/pitch/yaw [deg] with ±2σ covariance envelope
 *   • NIS vs χ²(m) threshold line
 *   • Position error per axis vs ±2σ bound (only when GT is recorded)
 *   • 3D / horizontal error magnitude    (only when GT is recorded)
 *
 * Requirements for gnuplot methods: gnuplot ≥ 5 in PATH.
 * toCSV / toMatlab have no runtime dependencies beyond <fstream>.
 *
 * Typical usage:
 *   kf::FilterRecorder rec;
 *   while (running) {
 *       filter.predict(imu, dt);
 *       double nis = filter.updatePosition(z_gnss, R_gnss);
 *       rec.record(t, filter, nis);           // basic
 *       rec.recordWithGT(t, filter, p_gt, nis); // with ground truth
 *       rec.updateLivePlot();                  // refresh live window
 *   }
 *   rec.toCSV("run.csv");
 *   rec.toMatlab("run.m");
 *   rec.showPlot();                            // final blocking view
 */

namespace kf {

// ── Snapshot ──────────────────────────────────────────────────────────────────

struct FilterSnapshot {
    double t              = 0.0;
    Eigen::Vector3d p     = Eigen::Vector3d::Zero();   // position   [m]
    Eigen::Vector3d v     = Eigen::Vector3d::Zero();   // velocity   [m/s]
    Eigen::Vector3d euler = Eigen::Vector3d::Zero();   // roll/pitch/yaw [rad]
    Eigen::Matrix<double, 9, 9> P = Eigen::Matrix<double, 9, 9>::Zero();
    double nis            = 0.0;
    bool   has_gt         = false;
    Eigen::Vector3d gt_p  = Eigen::Vector3d::Zero();   // ground-truth position [m]
};

// ── FilterRecorder ────────────────────────────────────────────────────────────

class FilterRecorder {
public:

    // ── Collection ────────────────────────────────────────────────────────────

    /** Record one filter snapshot at time t. nis is the most recent update's NIS. */
    void record(double t, const ESEKF& esekf, double nis = 0.0) {
        FilterSnapshot s;
        s.t     = t;
        s.p     = esekf.position();
        s.v     = esekf.velocity();
        s.euler = esekf.orientation().toEuler();   // roll, pitch, yaw [rad]
        s.P     = esekf.covariance();
        s.nis   = nis;
        history_.push_back(s);
    }

    /** Record with a ground-truth position for per-axis error and magnitude plots. */
    void recordWithGT(double t, const ESEKF& esekf,
                      const Eigen::Vector3d& gt_p, double nis = 0.0)
    {
        record(t, esekf, nis);
        history_.back().has_gt = true;
        history_.back().gt_p   = gt_p;
    }

    void   clear()                                       { history_.clear(); }
    size_t size()                                  const { return history_.size(); }
    const std::vector<FilterSnapshot>& history()   const { return history_; }

    // ── CSV export ────────────────────────────────────────────────────────────

    /**
     * Writes a comma-separated file.  Columns:
     *   t, px, py, pz,
     *   vx, vy, vz,
     *   roll_deg, pitch_deg, yaw_deg,
     *   sig_px, sig_py, sig_pz,        (1-sigma  [m])
     *   sig_vx, sig_vy, sig_vz,        (1-sigma  [m/s])
     *   sig_roll_deg, sig_pitch_deg, sig_yaw_deg,
     *   nis,
     *   gt_px, gt_py, gt_pz            (0 when no GT was recorded)
     */
    void toCSV(const std::string& path) const {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("FilterRecorder::toCSV: cannot open " + path);

        f << "t,px,py,pz,vx,vy,vz,roll_deg,pitch_deg,yaw_deg,"
          << "sig_px,sig_py,sig_pz,sig_vx,sig_vy,sig_vz,"
          << "sig_roll_deg,sig_pitch_deg,sig_yaw_deg,"
          << "nis,gt_px,gt_py,gt_pz\n";

        f << std::fixed << std::setprecision(6);
        for (const auto& s : history_) {
            f << s.t << ','
              << s.p.x()  << ',' << s.p.y()  << ',' << s.p.z() << ','
              << s.v.x()  << ',' << s.v.y()  << ',' << s.v.z() << ','
              << deg(s.euler(0)) << ',' << deg(s.euler(1)) << ',' << deg(s.euler(2)) << ','
              << sig(s.P, 0) << ',' << sig(s.P, 1) << ',' << sig(s.P, 2) << ','
              << sig(s.P, 3) << ',' << sig(s.P, 4) << ',' << sig(s.P, 5) << ','
              << deg(sig(s.P, 6)) << ',' << deg(sig(s.P, 7)) << ',' << deg(sig(s.P, 8)) << ','
              << s.nis << ','
              << (s.has_gt ? s.gt_p.x() : 0.0) << ','
              << (s.has_gt ? s.gt_p.y() : 0.0) << ','
              << (s.has_gt ? s.gt_p.z() : 0.0) << '\n';
        }
    }

    // ── MATLAB / Octave export ────────────────────────────────────────────────

    /**
     * Writes a self-contained MATLAB / Octave script.
     *
     * The generated figure has 4 subplots (position, velocity, orientation, NIS)
     * extended to 6 when ground-truth was recorded (adds per-axis error and 3D
     * error magnitude).  All signal subplots include ±2σ shaded bands.
     *
     * Run in MATLAB:  >> run('run.m')
     * Run in Octave:  $ octave run.m
     *
     * @param path        Output file path (use .m extension).
     * @param fig_title   Figure window title.
     * @param nis_dof     Degrees of freedom for the NIS chi-squared line.
     * @param confidence  Chi-squared confidence level (0.90 / 0.95 / 0.99).
     */
    void toMatlab(const std::string& path,
                  const std::string& fig_title = "ES-EKF",
                  int    nis_dof    = 3,
                  double confidence = 0.95) const
    {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("FilterRecorder::toMatlab: cannot open " + path);

        const bool   has_gt    = anyGT();
        const double nis_thr   = chi2Threshold(nis_dof, confidence);
        const int    conf_pct  = static_cast<int>(confidence * 100);

        f << std::fixed << std::setprecision(6);

        // ── Header ──
        f << "% " << fig_title << "  —  generated by kf::FilterRecorder\n"
          << "% Compatible with MATLAB R2016a+ and GNU Octave 6+\n"
          << "close all; clf;\n"
          << "figure('Name', '" << fig_title << "', 'NumberTitle', 'off', "
          << "'Position', [80 80 1400 " << (has_gt ? 900 : 620) << "]);\n\n";

        // ── Data arrays ──
        writeMatVec(f, "t",          [](const FilterSnapshot& s){ return std::to_string(s.t); });
        writeMatMat(f, "p",          [](const FilterSnapshot& s){
            return vec3Str(s.p.x(), s.p.y(), s.p.z()); });
        writeMatMat(f, "v",          [](const FilterSnapshot& s){
            return vec3Str(s.v.x(), s.v.y(), s.v.z()); });
        writeMatMat(f, "euler_deg",  [](const FilterSnapshot& s){
            return vec3Str(deg(s.euler(0)), deg(s.euler(1)), deg(s.euler(2))); });
        writeMatMat(f, "sig_p",      [](const FilterSnapshot& s){
            return vec3Str(sig(s.P,0), sig(s.P,1), sig(s.P,2)); });
        writeMatMat(f, "sig_v",      [](const FilterSnapshot& s){
            return vec3Str(sig(s.P,3), sig(s.P,4), sig(s.P,5)); });
        writeMatMat(f, "sig_e_deg",  [](const FilterSnapshot& s){
            return vec3Str(deg(sig(s.P,6)), deg(sig(s.P,7)), deg(sig(s.P,8))); });
        writeMatVec(f, "nis",        [](const FilterSnapshot& s){ return std::to_string(s.nis); });
        f << "nis_thr = " << nis_thr << ";  "
          << "% chi2(" << nis_dof << ", " << conf_pct << "%)\n";

        if (has_gt) {
            writeMatMat(f, "gt_p", [](const FilterSnapshot& s){
                return s.has_gt ? vec3Str(s.gt_p.x(), s.gt_p.y(), s.gt_p.z())
                                : std::string("NaN NaN NaN"); });
        }
        f << "\n";

        // ── sigma_band helper ──
        f << "function sigma_band(t, vals, sigs, col)\n"
          << "  t_f  = [t, fliplr(t)];\n"
          << "  y_f  = [vals + 2*sigs, fliplr(vals - 2*sigs)];\n"
          << "  fill(t_f, y_f, col, 'FaceAlpha', 0.18, 'EdgeColor', 'none');\n"
          << "end\n\n";

        const int rows = has_gt ? 3 : 2;
        const int cols = 2;

        // ── Subplot 1: Position ──
        f << "subplot(" << rows << "," << cols << ",1); hold on; grid on; box on;\n";
        matSigBand(f, "t", "p(:,1)'", "sig_p(:,1)'", "[0.00 0.45 0.74]");
        matSigBand(f, "t", "p(:,2)'", "sig_p(:,2)'", "[0.85 0.33 0.10]");
        matSigBand(f, "t", "p(:,3)'", "sig_p(:,3)'", "[0.47 0.67 0.19]");
        f << "plot(t, p(:,1), 'Color', [0.00 0.45 0.74], 'LineWidth', 1.5, 'DisplayName', 'x');\n"
          << "plot(t, p(:,2), 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5, 'DisplayName', 'y');\n"
          << "plot(t, p(:,3), 'Color', [0.47 0.67 0.19], 'LineWidth', 1.5, 'DisplayName', 'z');\n";
        if (has_gt) {
            f << "plot(t, gt_p(:,1), '--', 'Color', [0.00 0.45 0.74], 'LineWidth', 1.0, 'DisplayName', 'gt x');\n"
              << "plot(t, gt_p(:,2), '--', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.0, 'DisplayName', 'gt y');\n"
              << "plot(t, gt_p(:,3), '--', 'Color', [0.47 0.67 0.19], 'LineWidth', 1.0, 'DisplayName', 'gt z');\n";
        }
        f << "title('Position'); xlabel('t [s]'); ylabel('[m]');\n"
          << "legend('Location', 'best'); hold off;\n\n";

        // ── Subplot 2: Velocity ──
        f << "subplot(" << rows << "," << cols << ",2); hold on; grid on; box on;\n";
        matSigBand(f, "t", "v(:,1)'", "sig_v(:,1)'", "[0.00 0.45 0.74]");
        matSigBand(f, "t", "v(:,2)'", "sig_v(:,2)'", "[0.85 0.33 0.10]");
        matSigBand(f, "t", "v(:,3)'", "sig_v(:,3)'", "[0.47 0.67 0.19]");
        f << "plot(t, v(:,1), 'Color', [0.00 0.45 0.74], 'LineWidth', 1.5, 'DisplayName', 'vx');\n"
          << "plot(t, v(:,2), 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5, 'DisplayName', 'vy');\n"
          << "plot(t, v(:,3), 'Color', [0.47 0.67 0.19], 'LineWidth', 1.5, 'DisplayName', 'vz');\n"
          << "title('Velocity'); xlabel('t [s]'); ylabel('[m/s]');\n"
          << "legend('Location', 'best'); hold off;\n\n";

        // ── Subplot 3: Euler angles ──
        f << "subplot(" << rows << "," << cols << ",3); hold on; grid on; box on;\n";
        matSigBand(f, "t", "euler_deg(:,1)'", "sig_e_deg(:,1)'", "[0.00 0.45 0.74]");
        matSigBand(f, "t", "euler_deg(:,2)'", "sig_e_deg(:,2)'", "[0.85 0.33 0.10]");
        matSigBand(f, "t", "euler_deg(:,3)'", "sig_e_deg(:,3)'", "[0.47 0.67 0.19]");
        f << "plot(t, euler_deg(:,1), 'Color', [0.00 0.45 0.74], 'LineWidth', 1.5, 'DisplayName', 'roll');\n"
          << "plot(t, euler_deg(:,2), 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5, 'DisplayName', 'pitch');\n"
          << "plot(t, euler_deg(:,3), 'Color', [0.47 0.67 0.19], 'LineWidth', 1.5, 'DisplayName', 'yaw');\n"
          << "title('Orientation (Euler)'); xlabel('t [s]'); ylabel('[deg]');\n"
          << "legend('Location', 'best'); hold off;\n\n";

        // ── Subplot 4: NIS ──
        f << "subplot(" << rows << "," << cols << ",4); hold on; grid on; box on;\n"
          << "plot(t, nis, 'Color', [0.00 0.45 0.74], 'LineWidth', 1.5, 'DisplayName', 'NIS');\n"
          << "plot([t(1), t(end)], [nis_thr, nis_thr], '--r', 'LineWidth', 1.5, "
          << "'DisplayName', sprintf('chi2(" << nis_dof << ", " << conf_pct << "%%): %.2f', nis_thr));\n"
          << "title('NIS'); xlabel('t [s]'); ylabel('NIS');\n"
          << "legend('Location', 'best'); hold off;\n\n";

        if (has_gt) {
            // ── Subplot 5: Per-axis position error vs ±2σ ──
            f << "err = p - gt_p;\n\n";
            f << "subplot(" << rows << "," << cols << ",5); hold on; grid on; box on;\n";
            matSigBand(f, "t", "err(:,1)'", "sig_p(:,1)'", "[0.00 0.45 0.74]");
            matSigBand(f, "t", "err(:,2)'", "sig_p(:,2)'", "[0.85 0.33 0.10]");
            matSigBand(f, "t", "err(:,3)'", "sig_p(:,3)'", "[0.47 0.67 0.19]");
            f << "plot(t, err(:,1), 'Color', [0.00 0.45 0.74], 'LineWidth', 1.5, 'DisplayName', 'err x');\n"
              << "plot(t, err(:,2), 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5, 'DisplayName', 'err y');\n"
              << "plot(t, err(:,3), 'Color', [0.47 0.67 0.19], 'LineWidth', 1.5, 'DisplayName', 'err z');\n"
              << "title('Position Error vs Ground Truth (\\pm2\\sigma)'); xlabel('t [s]'); ylabel('[m]');\n"
              << "legend('Location', 'best'); hold off;\n\n";

            // ── Subplot 6: 3D & horizontal error magnitude ──
            f << "subplot(" << rows << "," << cols << ",6); hold on; grid on; box on;\n"
              << "plot(t, sqrt(sum(err.^2, 2)), 'k', 'LineWidth', 1.5, 'DisplayName', '3D');\n"
              << "plot(t, sqrt(err(:,1).^2 + err(:,2).^2), 'Color', [0.85 0.33 0.10], "
              << "'LineWidth', 1.2, 'DisplayName', 'horiz');\n"
              << "title('Position Error Magnitude'); xlabel('t [s]'); ylabel('[m]');\n"
              << "legend('Location', 'best'); hold off;\n\n";
        }

        f << "sgtitle('" << fig_title << "');\n";
    }

    // ── Gnuplot ───────────────────────────────────────────────────────────────

    /**
     * Opens a blocking gnuplot window showing all subplots.
     * Returns after the window is closed.
     * @param nis_dof     DOF for the chi-squared NIS threshold line.
     * @param confidence  Confidence level (0.90 / 0.95 / 0.99).
     */
    void showPlot(int nis_dof = 3, double confidence = 0.95) const {
        if (history_.empty()) return;
        FILE* gp = popen("gnuplot --persist", "w");
        if (!gp) throw std::runtime_error("FilterRecorder::showPlot: gnuplot not found in PATH");
        sendToGnuplot(gp, nis_dof, confidence);
        fflush(gp);
        pclose(gp);
    }

    /**
     * Opens (on first call) and refreshes a live gnuplot window.
     * Call once per filter cycle after record().  The pipe stays open
     * between calls; call closeLivePlot() when done.
     */
    void updateLivePlot(int nis_dof = 3, double confidence = 0.95) {
        if (history_.empty()) return;
        if (!live_pipe_) {
            live_pipe_ = popen("gnuplot", "w");
            if (!live_pipe_)
                throw std::runtime_error("FilterRecorder::updateLivePlot: gnuplot not found");
        }
        sendToGnuplot(live_pipe_, nis_dof, confidence);
        fflush(live_pipe_);
    }

    void closeLivePlot() {
        if (live_pipe_) { pclose(live_pipe_); live_pipe_ = nullptr; }
    }

    ~FilterRecorder() { closeLivePlot(); }

private:
    std::vector<FilterSnapshot> history_;
    FILE* live_pipe_ = nullptr;

    // ── Scalar helpers ────────────────────────────────────────────────────────

    static double deg(double rad) { return rad * (180.0 / M_PI); }
    static double sig(const Eigen::Matrix<double, 9, 9>& P, int i) {
        return std::sqrt(P(i, i));
    }

    bool anyGT() const {
        return std::any_of(history_.begin(), history_.end(),
                           [](const FilterSnapshot& s){ return s.has_gt; });
    }

    // ── MATLAB codegen helpers ────────────────────────────────────────────────

    static std::string vec3Str(double a, double b, double c) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(6) << a << ' ' << b << ' ' << c;
        return ss.str();
    }

    // Write: name = [v1; v2; ...];   (row vector)
    void writeMatVec(std::ofstream& f, const std::string& name,
                     std::function<std::string(const FilterSnapshot&)> fn) const
    {
        f << name << " = [";
        for (size_t i = 0; i < history_.size(); ++i) {
            if (i) f << ", ";
            f << fn(history_[i]);
        }
        f << "];\n";
    }

    // Write: name = [row1; row2; ...];   (matrix, 3 columns)
    void writeMatMat(std::ofstream& f, const std::string& name,
                     std::function<std::string(const FilterSnapshot&)> fn) const
    {
        f << name << " = [";
        for (size_t i = 0; i < history_.size(); ++i) {
            if (i) f << "; ";
            f << fn(history_[i]);
        }
        f << "];\n";
    }

    static void matSigBand(std::ofstream& f,
                           const std::string& t,
                           const std::string& vals,
                           const std::string& sigs,
                           const std::string& col)
    {
        f << "sigma_band(" << t << ", " << vals << ", " << sigs << ", " << col << ");\n";
    }

    // ── gnuplot codegen ───────────────────────────────────────────────────────

    // Write inline dataset:  t  val  sigma  (terminated by 'e')
    // col: 0-2 → position, 3-5 → velocity, 6-8 → euler (deg)
    void writeGpChannel(FILE* gp, int col) const {
        for (const auto& s : history_) {
            double val, sigma;
            if (col < 3) {
                val = s.p(col);  sigma = sig(s.P, col);
            } else if (col < 6) {
                val = s.v(col-3);  sigma = sig(s.P, col);
            } else {
                val = deg(s.euler(col-6));  sigma = deg(sig(s.P, col));
            }
            fprintf(gp, "%f %f %f\n", s.t, val, sigma);
        }
        fprintf(gp, "e\n");
    }

    // Write inline NIS dataset:  t  nis  (terminated by 'e')
    void writeGpNis(FILE* gp) const {
        for (const auto& s : history_) fprintf(gp, "%f %f\n", s.t, s.nis);
        fprintf(gp, "e\n");
    }

    // Write inline GT-error dataset:  t  ex  ey  ez  sig_x  sig_y  sig_z
    void writeGpGtErr(FILE* gp) const {
        for (const auto& s : history_) {
            const Eigen::Vector3d e = s.has_gt ? Eigen::Vector3d(s.p - s.gt_p)
                                                            : Eigen::Vector3d::Zero();
            fprintf(gp, "%f %f %f %f %f %f %f\n",
                    s.t, e.x(), e.y(), e.z(),
                    sig(s.P,0), sig(s.P,1), sig(s.P,2));
        }
        fprintf(gp, "e\n");
    }

    // Emit a 3-channel subplot (sigma bands + lines) with 6 inline datasets.
    // col_base: 0=position, 3=velocity, 6=euler
    void gpThreeChannel(FILE* gp, const char* title, const char* ylabel,
                        const char* la, const char* lb, const char* lc,
                        int col_base) const
    {
        static const char* kC[] = { "#0072BD", "#D95319", "#77AC30" };
        fprintf(gp,
            "set title '%s'\nset ylabel '%s'\n"
            "plot '-' u 1:($2-2*$3):($2+2*$3) w filledcurves lc rgb '%s' notitle,"
            "     '-' u 1:($2-2*$3):($2+2*$3) w filledcurves lc rgb '%s' notitle,"
            "     '-' u 1:($2-2*$3):($2+2*$3) w filledcurves lc rgb '%s' notitle,"
            "     '-' u 1:2 w lines lc rgb '%s' lw 2 title '%s',"
            "     '-' u 1:2 w lines lc rgb '%s' lw 2 title '%s',"
            "     '-' u 1:2 w lines lc rgb '%s' lw 2 title '%s'\n",
            title, ylabel,
            kC[0], kC[1], kC[2],
            kC[0], la, kC[1], lb, kC[2], lc);
        for (int i = 0; i < 6; ++i) writeGpChannel(gp, col_base + i % 3);
    }

    void sendToGnuplot(FILE* gp, int nis_dof, double confidence) const {
        const bool   has_gt   = anyGT();
        const double nis_thr  = chi2Threshold(nis_dof, confidence);
        const int    conf_pct = static_cast<int>(confidence * 100);
        const int    rows     = has_gt ? 3 : 2;

        fprintf(gp,
            "set terminal wxt size 1400,%d enhanced font 'Arial,10'\n"
            "set style fill transparent solid 0.25 noborder\n"
            "set multiplot layout %d,2 title 'ES-EKF Filter State'\n"
            "set grid\nset xlabel 't [s]'\nset key top right\n",
            has_gt ? 900 : 660, rows);

        gpThreeChannel(gp, "Position [m]",           "[m]",    "x",    "y",    "z",    0);
        gpThreeChannel(gp, "Velocity [m/s]",          "[m/s]",  "vx",   "vy",   "vz",   3);
        gpThreeChannel(gp, "Orientation (Euler) [deg]","[deg]","roll","pitch", "yaw",   6);

        // NIS subplot
        fprintf(gp,
            "set title 'NIS'\nset ylabel 'NIS'\n"
            "plot '-' u 1:2 w lines lc rgb '#0072BD' lw 2 title 'NIS',"
            "     %f w lines lc rgb 'red' lw 2 dt 2 title 'chi2(%d, %d%%)'\n",
            nis_thr, nis_dof, conf_pct);
        writeGpNis(gp);

        if (has_gt) {
            // Per-axis error vs ±2σ
            static const char* kC[] = { "#0072BD", "#D95319", "#77AC30" };
            fprintf(gp,
                "set title 'Position Error vs GT (\\261 2\\163)'\nset ylabel '[m]'\n"
                "plot '-' u 1:($2-2*$5):($2+2*$5) w filledcurves lc rgb '%s' notitle,"
                "     '-' u 1:($3-2*$6):($3+2*$6) w filledcurves lc rgb '%s' notitle,"
                "     '-' u 1:($4-2*$7):($4+2*$7) w filledcurves lc rgb '%s' notitle,"
                "     '-' u 1:2 w lines lc rgb '%s' lw 2 title 'err x',"
                "     '-' u 1:3 w lines lc rgb '%s' lw 2 title 'err y',"
                "     '-' u 1:4 w lines lc rgb '%s' lw 2 title 'err z'\n",
                kC[0], kC[1], kC[2], kC[0], kC[1], kC[2]);
            for (int i = 0; i < 6; ++i) writeGpGtErr(gp);

            // 3D / horizontal error magnitude
            fprintf(gp,
                "set title '3D Position Error Magnitude'\nset ylabel '[m]'\n"
                "plot '-' u 1:(sqrt($2*$2+$3*$3+$4*$4)) w lines lc rgb '#7E2F8E' lw 2 title '3D',"
                "     '-' u 1:(sqrt($2*$2+$3*$3))        w lines lc rgb '%s'      lw 2 title 'horiz'\n",
                kC[1]);
            writeGpGtErr(gp);
            writeGpGtErr(gp);
        }

        fprintf(gp, "unset multiplot\n");
    }
};

} // namespace kf

#endif // KF_GRAPH_UTILS_H
