"""
综合分析脚本
对比不同模型、注意力机制、PEFT方法的效果
生成全面的可视化图表和深度分析报告
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import config
from sklearn.metrics import auc
import warnings

warnings.filterwarnings("ignore")

# 设置绘图风格
plt.style.use(config.plot_style)
plt.rcParams["figure.dpi"] = config.dpi
plt.rcParams["figure.figsize"] = config.figure_size


class ComprehensiveAnalyzer:
    """综合分析器"""

    def __init__(self):
        self.results = {}
        self.models = ["bert", "roberta", "electra"]
        self.attention_types = ["MHA", "MGA", "GQA", "MQA"]
        self.peft_methods = ["lora", "prefix_tuning", "prompt_tuning"]

    def load_all_results(self):
        """加载所有训练结果"""
        print("加载训练结果...")

        for model in self.models:
            result_path = f"{config.results_dir}/{model}/train_results.pkl"
            if os.path.exists(result_path):
                with open(result_path, "rb") as f:
                    self.results[model] = pickle.load(f)
                print(f"  已加载 {model} 的结果")
            else:
                print(f"  警告: {model} 的结果文件不存在")

    def load_inference_results(self):
        """加载推理结果"""
        print("加载推理结果...")

        for model in self.models:
            result_path = f"{config.inference_results_dir}/{model}_inference.pkl"
            if os.path.exists(result_path):
                with open(result_path, "rb") as f:
                    self.results[f"{model}_inference"] = pickle.load(f)
                print(f"  已加载 {model} 的推理结果")

    def create_comparison_table(self):
        """创建模型对比表格"""
        print("\n生成模型对比表格...")

        metrics = ["accuracy", "f1", "precision", "recall", "auc", "mcc"]
        rows = []

        for model in self.models:
            if model not in self.results:
                continue

            data = self.results[model]
            train_results = data.get("train_results", {})
            eval_results = data.get("eval_results", {})

            row = {"model": model}
            for metric in metrics:
                train_key = f"train_{metric}"
                eval_key = f"eval_{metric}"

                train_val = train_results.get(train_key, "N/A")
                eval_val = eval_results.get(eval_key, "N/A")

                if train_val != "N/A" and eval_val != "N/A":
                    row[f"train_{metric}"] = f"{train_val:.4f}"
                    row[f"eval_{metric}"] = f"{eval_val:.4f}"
                    row[f"gap_{metric}"] = f"{train_val - eval_val:+.4f}"
                else:
                    row[f"train_{metric}"] = str(train_val)
                    row[f"eval_{metric}"] = str(eval_val)
                    row[f"gap_{metric}"] = "N/A"

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def plot_metrics_comparison(self):
        """绘制指标对比图"""
        print("生成指标对比图...")

        metrics = ["accuracy", "f1", "precision", "recall", "auc"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            models = []
            train_vals = []
            eval_vals = []

            for model in self.models:
                if model not in self.results:
                    continue

                data = self.results[model]
                train_val = data.get("train_results", {}).get(f"train_{metric}", None)
                eval_val = data.get("eval_results", {}).get(f"eval_{metric}", None)

                if train_val is not None and eval_val is not None:
                    models.append(model)
                    train_vals.append(train_val)
                    eval_vals.append(eval_val)

            x = np.arange(len(models))
            width = 0.35

            ax.bar(x - width / 2, train_vals, width, label="训练集", alpha=0.8)
            ax.bar(x + width / 2, eval_vals, width, label="验证集", alpha=0.8)

            ax.set_xlabel("模型")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} 对比")
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 隐藏第6个子图
        axes[5].axis("off")

        plt.suptitle("模型性能对比 - 训练集 vs 验证集", fontsize=16, y=1.02)
        plt.tight_layout()

        save_path = f"{config.visualization_dir}/metrics_comparison.png"
        plt.savefig(save_path, format=config.save_format, bbox_inches="tight")
        plt.close()
        print(f"  已保存到: {save_path}")

    def plot_learning_curves(self):
        """绘制学习曲线"""
        print("生成学习曲线...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        for idx, model in enumerate(self.models):
            if model not in self.results or idx >= 4:
                continue

            ax = axes[idx]
            data = self.results[model]

            # 损失曲线
            loss_history = data.get("loss_history", [])
            if loss_history:
                steps = [h["step"] for h in loss_history]
                losses = [h["loss"] for h in loss_history]
                ax.plot(steps, losses, label="训练损失", alpha=0.7)

            # 评估指标曲线
            eval_history = data.get("eval_metrics_history", [])
            if eval_history:
                steps = [h["step"] for h in eval_history]
                f1_scores = [h.get("f1", 0) for h in eval_history]
                ax_twin = ax.twinx()
                ax_twin.plot(steps, f1_scores, "r--", label="F1分数", alpha=0.7)
                ax_twin.set_ylabel("F1分数", color="r")
                ax_twin.tick_params(axis="y", labelcolor="r")

            ax.set_xlabel("训练步数")
            ax.set_ylabel("损失")
            ax.set_title(f"{model} - 学习曲线")
            ax.grid(True, alpha=0.3)

            # 合并图例
            lines1, labels1 = ax.get_legend_handles_labels()
            if eval_history:
                lines2, labels2 = ax_twin.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
            else:
                ax.legend()

        plt.suptitle("学习曲线对比", fontsize=16, y=1.02)
        plt.tight_layout()

        save_path = f"{config.visualization_dir}/learning_curves.png"
        plt.savefig(save_path, format=config.save_format, bbox_inches="tight")
        plt.close()
        print(f"  已保存到: {save_path}")

    def plot_confusion_matrices(self):
        """绘制所有模型的混淆矩阵"""
        print("生成混淆矩阵对比...")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, model in enumerate(self.models):
            ax = axes[idx]

            # 从推理结果中获取混淆矩阵
            inference_key = f"{model}_inference"
            if inference_key in self.results:
                cm = np.array(
                    self.results[inference_key].get(
                        "confusion_matrix", [[0, 0], [0, 0]]
                    )
                )
            else:
                # 否则从训练结果中获取
                if model in self.results:
                    eval_results = self.results[model].get("eval_results", {})
                    tp = eval_results.get("eval_true_positives", 0)
                    tn = eval_results.get("eval_true_negatives", 0)
                    fp = eval_results.get("eval_false_positives", 0)
                    fn = eval_results.get("eval_false_negatives", 0)
                    cm = np.array([[tn, fp], [fn, tp]])
                else:
                    cm = np.array([[0, 0], [0, 0]])

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["负类", "正类"],
                yticklabels=["负类", "正类"],
            )
            ax.set_title(f"{model} - 混淆矩阵")
            ax.set_ylabel("真实标签")
            ax.set_xlabel("预测标签")

        plt.tight_layout()

        save_path = f"{config.visualization_dir}/confusion_matrices_comparison.png"
        plt.savefig(save_path, format=config.save_format, bbox_inches="tight")
        plt.close()
        print(f"  已保存到: {save_path}")

    def plot_roc_curves_comparison(self):
        """绘制ROC曲线对比"""
        print("生成ROC曲线对比...")

        plt.figure(figsize=(10, 8))

        for model in self.models:
            inference_key = f"{model}_inference"
            if inference_key in self.results:
                data = self.results[inference_key]
                true_labels = data.get("true_labels", [])
                probabilities = data.get("probabilities", [])

                if true_labels and probabilities:
                    from sklearn.metrics import roc_curve

                    positive_probs = [p[1] for p in probabilities]
                    fpr, tpr, _ = roc_curve(true_labels, positive_probs)
                    auc_score = data["performance_metrics"].get("auc", 0)

                    plt.plot(fpr, tpr, label=f"{model} (AUC = {auc_score:.4f})", lw=2)

        plt.plot([0, 1], [0, 1], "k--", label="随机分类器", lw=2)
        plt.xlabel("假阳性率")
        plt.ylabel("真阳性率")
        plt.title("ROC曲线对比")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = f"{config.visualization_dir}/roc_curves_comparison.png"
        plt.savefig(save_path, format=config.save_format, bbox_inches="tight")
        plt.close()
        print(f"  已保存到: {save_path}")

    def plot_performance_radar(self):
        """绘制性能雷达图"""
        print("生成性能雷达图...")

        from math import pi

        metrics = ["accuracy", "f1", "precision", "recall", "auc"]
        N = len(metrics)

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # 闭合

        colors = ["blue", "red", "green"]

        for idx, model in enumerate(self.models):
            if model not in self.results:
                continue

            values = []
            data = self.results[model]
            eval_results = data.get("eval_results", {})

            for metric in metrics:
                val = eval_results.get(f"eval_{metric}", 0)
                values.append(val)

            values += values[:1]  # 闭合

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=model.upper(),
                color=colors[idx],
            )
            ax.fill(angles, values, alpha=0.1, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title("模型性能雷达图", size=16, y=1.1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        save_path = f"{config.visualization_dir}/performance_radar.png"
        plt.savefig(save_path, format=config.save_format, bbox_inches="tight")
        plt.close()
        print(f"  已保存到: {save_path}")

    def plot_memory_and_time_analysis(self):
        """绘制内存和时间分析图"""
        print("生成内存和时间分析图...")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        models = []
        peak_memories = []
        train_times = []
        throughputs = []

        for model in self.models:
            if model not in self.results:
                continue

            data = self.results[model]
            models.append(model)
            peak_memories.append(data.get("peak_memory", 0))
            train_times.append(data.get("total_train_time", 0))

            # 计算吞吐量
            total_tokens = data.get("total_tokens", 1)
            total_time = data.get("total_train_time", 1)
            throughputs.append(total_tokens / total_time if total_time > 0 else 0)

        # 峰值内存
        axes[0].bar(models, peak_memories, color=["blue", "red", "green"])
        axes[0].set_ylabel("峰值内存 (GB)")
        axes[0].set_title("训练峰值内存对比")
        axes[0].grid(True, alpha=0.3)

        # 训练时间
        axes[1].bar(models, train_times, color=["blue", "red", "green"])
        axes[1].set_ylabel("训练时间 (秒)")
        axes[1].set_title("训练时间对比")
        axes[1].grid(True, alpha=0.3)

        # 吞吐量
        axes[2].bar(models, throughputs, color=["blue", "red", "green"])
        axes[2].set_ylabel("吞吐量 (tokens/s)")
        axes[2].set_title("训练吞吐量对比")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = f"{config.visualization_dir}/memory_time_analysis.png"
        plt.savefig(save_path, format=config.save_format, bbox_inches="tight")
        plt.close()
        print(f"  已保存到: {save_path}")

    def generate_analysis_report(self):
        """生成深度分析报告"""
        print("\n生成深度分析报告...")

        report = []
        report.append("=" * 80)
        report.append("文本分类模型综合分析报告")
        report.append("=" * 80)
        report.append("")

        # 1. 模型性能总览
        report.append("1. 模型性能总览")
        report.append("-" * 80)
        df = self.create_comparison_table()
        report.append(df.to_string(index=False))
        report.append("")

        # 2. 过拟合分析
        report.append("2. 过拟合分析 (训练集-验证集差异)")
        report.append("-" * 80)

        for model in self.models:
            if model not in self.results:
                continue

            data = self.results[model]
            train_results = data.get("train_results", {})
            eval_results = data.get("eval_results", {})

            report.append(f"\n{model.upper()} 模型:")

            for metric in ["accuracy", "f1", "precision", "recall"]:
                train_val = train_results.get(f"train_{metric}", None)
                eval_val = eval_results.get(f"eval_{metric}", None)

                if train_val is not None and eval_val is not None:
                    gap = train_val - eval_val
                    report.append(
                        f"  {metric.upper()}: 训练={train_val:.4f}, 验证={eval_val:.4f}, 差异={gap:+.4f}"
                    )

                    if abs(gap) > 0.1:
                        report.append(f"    ⚠️  警告: {metric} 差异较大，可能存在过拟合")

        report.append("")

        # 3. 训练效率分析
        report.append("3. 训练效率分析")
        report.append("-" * 80)

        for model in self.models:
            if model not in self.results:
                continue

            data = self.results[model]
            report.append(f"\n{model.upper()} 模型:")
            report.append(f"  训练时间: {data.get('total_train_time', 0):.2f} 秒")
            report.append(f"  峰值内存: {data.get('peak_memory', 0):.2f} GB")
            report.append(f"  总Token数: {data.get('total_tokens', 0)}")

            total_time = data.get("total_train_time", 1)
            total_tokens = data.get("total_tokens", 1)
            report.append(f"  吞吐量: {total_tokens / total_time:.2f} tokens/s")

        report.append("")

        # 4. 推理性能分析（如果有）
        report.append("4. 推理性能分析")
        report.append("-" * 80)

        for model in self.models:
            inference_key = f"{model}_inference"
            if inference_key in self.results:
                data = self.results[inference_key]
                perf = data.get("performance_metrics", {})

                report.append(f"\n{model.upper()} 模型:")
                report.append(
                    f"  推理吞吐量: {perf.get('throughput_samples', 0):.2f} samples/s"
                )
                report.append(
                    f"  Token吞吐量: {perf.get('throughput_tokens', 0):.2f} tokens/s"
                )
                report.append(
                    f"  平均推理时间: {perf.get('avg_time_per_sample', 0) * 1000:.2f} ms"
                )
                report.append(f"  峰值GPU显存: {perf.get('peak_memory_gb', 0):.2f} GB")

        report.append("")
        report.append("=" * 80)
        report.append("报告生成完成")
        report.append("=" * 80)

        # 保存报告
        report_text = "\n".join(report)
        report_path = f"{config.results_dir}/comprehensive_analysis_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(f"\n报告已保存到: {report_path}")
        print("\n" + report_text)

    def run_full_analysis(self):
        """运行完整分析"""
        print("\n" + "=" * 80)
        print("开始综合分析")
        print("=" * 80)

        # 创建目录
        os.makedirs(config.visualization_dir, exist_ok=True)

        # 加载结果
        self.load_all_results()
        self.load_inference_results()

        if not self.results:
            print("错误: 没有找到任何结果文件。请先运行训练。")
            return

        # 生成各种图表
        self.plot_metrics_comparison()
        self.plot_learning_curves()
        self.plot_confusion_matrices()
        self.plot_roc_curves_comparison()
        self.plot_performance_radar()
        self.plot_memory_and_time_analysis()

        # 生成报告
        self.generate_analysis_report()

        # 保存所有图表到PDF
        print("\n将所有图表合并到PDF...")
        pdf_path = f"{config.visualization_dir}/comprehensive_analysis.pdf"

        with PdfPages(pdf_path) as pdf:
            for filename in os.listdir(config.visualization_dir):
                if filename.endswith(".png"):
                    img_path = os.path.join(config.visualization_dir, filename)
                    img = plt.imread(img_path)
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.imshow(img)
                    ax.axis("off")
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close()

        print(f"PDF报告已保存到: {pdf_path}")
        print("\n" + "=" * 80)
        print("综合分析完成!")
        print("=" * 80)


if __name__ == "__main__":
    analyzer = ComprehensiveAnalyzer()
    analyzer.run_full_analysis()
