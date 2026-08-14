#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MC 史莱姆区块种子搜索器 - 管理界面（tkinter）

功能：
  - 显示/编辑计算参数（种子范围、checkpoint；窗口/阈值/网格为编译期常量只读显示）
  - 开始/停止开关（子进程跑 CUDA 搜索，GUI 零计算负担）
  - 底部输出框：只在「找到种子」时输出结果；找不到静默（状态栏提示），
    避免大量 IO 影响 GPU 搜索效率
  - 进度显示：每 1s 低频读 checkpoint 文件（一次小文件读，开销可忽略）

用法：
  py gui.py
"""
import os
import re
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(BASE_DIR, "gpu", "slime_search_bp.exe")
DEFAULT_CP = os.path.join(BASE_DIR, "cp_gui.dat")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 与 gpu/slime_search_bp.cu 一致的编译期常量（只读显示）
WINDOW, THRESHOLD, GRID = 5, 19, 401

RE_FOUND = re.compile(r"seed:\s*(-?\d+)")   # -? 支持负方向（负种子）
RE_WIN = re.compile(r"window top-left chunk:\s*\((-?\d+),\s*(-?\d+)\)")
RE_COUNT = re.compile(r"5x5 slime count:\s*(\d+)")
RE_NOFOUND = re.compile(r"no dense seed found")
RE_THROUGHPUT = re.compile(r"throughput:\s*([\d.]+)")


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.proc = None
        self.start_time = 0.0
        self.checked = 0
        self.running = False

        root.title("MC 史莱姆区块种子搜索器")
        root.geometry("680x520")
        root.minsize(560, 460)

        # ---- 参数区 ----
        frm = ttk.LabelFrame(root, text="计算参数", padding=8)
        frm.pack(fill="x", padx=10, pady=6)

        row = ttk.Frame(frm)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="种子范围 start:").pack(side="left")
        self.var_start = tk.StringVar(value="0")
        ttk.Entry(row, textvariable=self.var_start, width=14).pack(side="left", padx=4)
        ttk.Label(row, text="end:").pack(side="left")
        self.var_end = tk.StringVar(value="")   # 留空 = 无限搜索
        ttk.Entry(row, textvariable=self.var_end, width=14).pack(side="left", padx=4)
        ttk.Label(row, text="(留空=一直算不停)", foreground="#888").pack(side="left")

        row = ttk.Frame(frm)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="方向:").pack(side="left")
        self.var_dir = tk.StringVar(value="正")
        ttk.Radiobutton(row, text="正方向（0,1,2,...）", variable=self.var_dir,
                        value="正", command=self._on_dir_change).pack(side="left", padx=4)
        ttk.Radiobutton(row, text="负方向（0,-1,-2,...）", variable=self.var_dir,
                        value="负", command=self._on_dir_change).pack(side="left", padx=4)

        row = ttk.Frame(frm)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="阈值 n (≥):").pack(side="left")
        self.var_threshold = tk.StringVar(value="19")
        ttk.Entry(row, textvariable=self.var_threshold, width=8).pack(side="left", padx=4)
        ttk.Label(row, text="(窗口内史莱姆区块数达到 n 即命中，1-25)", foreground="#888").pack(side="left")

        row = ttk.Frame(frm)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="checkpoint 文件:").pack(side="left")
        self.var_cp = tk.StringVar(value=DEFAULT_CP)
        ttk.Entry(row, textvariable=self.var_cp).pack(side="left", fill="x", expand=True, padx=4)

        row = ttk.Frame(frm)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="窗口: 5×5 (编译期)   网格: 401×401 (编译期)",
                  foreground="#666").pack(side="left")
        ttk.Label(row, text=f"GPU: {os.path.basename(EXE)}", foreground="#666").pack(side="right")

        # ---- 控制区 ----
        frm2 = ttk.LabelFrame(root, text="控制", padding=8)
        frm2.pack(fill="x", padx=10, pady=6)

        ctl = ttk.Frame(frm2)
        ctl.pack(fill="x")
        self.btn_start = ttk.Button(ctl, text="▶ 开始", command=self.start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(ctl, text="⏹ 停止", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=8)
        self.var_pause = tk.BooleanVar(value=True)   # 默认：找到一个就暂停
        ttk.Checkbutton(ctl, text="找到一个就暂停", variable=self.var_pause).pack(side="left", padx=8)

        self.var_status = tk.StringVar(value="状态: 空闲")
        ttk.Label(frm2, textvariable=self.var_status).pack(anchor="w", pady=(6, 0))
        self.var_progress = tk.StringVar(value="进度: -")
        ttk.Label(frm2, textvariable=self.var_progress).pack(anchor="w")

        # ---- 输出区（只在找到种子时写） ----
        frm3 = ttk.LabelFrame(root, text="日志（开始/暂停/结果；搜索过程中不刷屏）", padding=8)
        frm3.pack(fill="both", expand=True, padx=10, pady=6)
        self.txt = tk.Text(frm3, height=8, wrap="word", state="disabled",
                           font=("Consolas", 10))
        self.txt.pack(fill="both", expand=True)
        sb = ttk.Scrollbar(self.txt, command=self.txt.yview)
        sb.pack(side="right", fill="y")
        self.txt.configure(yscrollcommand=sb.set)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._poll)   # 0.1s 刷新（文件读在 page cache，零 GPU 影响）

    # ---------- 日志文件（一个开始→暂停/找到周期 = 一个文件，事件全部写入） ----------
    def _log_name_parts(self):
        """日志文件名要素：(start, end字符串, 阈值, 方向)"""
        try:
            start = int(self.var_start.get())
            threshold = int(self.var_threshold.get())
        except ValueError:
            start, threshold = 0, 0
        end_text = self.var_end.get().strip()
        end_str = end_text if end_text else "Inf"
        dir_str = "neg" if self.var_dir.get() == "负" else "pos"
        return start, end_str, threshold, dir_str

    def _init_log(self, lines: list):
        """开始搜索时创建会话日志文件（文件名带时间戳+start/end+n+方向，状态为『进行中』）"""
        start, end_str, threshold, dir_str = self._log_name_parts()
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            self.log_ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"log_{self.log_ts}_s{start}_e{end_str}_n{threshold}_{dir_str}_进行中.txt"
            self.log_path = os.path.join(LOG_DIR, fname)
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError:
            self.log_path = None

    def _append_log(self, lines: list):
        if not getattr(self, "log_path", None):
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError:
            pass

    def _rename_log(self, status: str):
        """会话结束（暂停/找到/未找到）时把文件重命名为最终状态"""
        if not getattr(self, "log_path", None) or not getattr(self, "log_ts", None):
            return
        start, end_str, threshold, dir_str = self._log_name_parts()
        try:
            new = os.path.join(LOG_DIR, f"log_{self.log_ts}_s{start}_e{end_str}_n{threshold}_{dir_str}_{status}.txt")
            if os.path.isfile(self.log_path):
                os.rename(self.log_path, new)
            self.log_path = new
        except OSError:
            pass

    # ---------- 控制 ----------
    def _on_dir_change(self):
        """切方向时防误操作：负方向 + 正数 start = 搜正数区间（易误以为方向没生效）"""
        if self.var_dir.get() != "负":
            return
        try:
            start = int(self.var_start.get())
        except ValueError:
            return
        if start > 0:
            if messagebox.askyesno(
                    "方向切换提示",
                    f"负方向通常从 0 开始往负数搜（0,-1,-2,...）。\n"
                    f"当前 start={start:,} 是正数，将从该值往下搜索【正数区间】。\n\n"
                    f"要把 start 重置为 0 吗？"):
                self.var_start.set("0")

    def start(self):
        if self.running:
            return
        try:
            start = int(self.var_start.get())
            threshold = int(self.var_threshold.get())
        except ValueError:
            messagebox.showerror("参数错误", "start / 阈值 必须是整数")
            return
        neg = (self.var_dir.get() == "负")
        end_text = self.var_end.get().strip()
        if end_text:
            try:
                end = int(end_text)
            except ValueError:
                messagebox.showerror("参数错误", "end 必须是整数或留空（无限）")
                return
            if neg and end >= start:
                messagebox.showerror("参数错误", "负方向 end 必须 < start")
                return
            if not neg and end <= start:
                messagebox.showerror("参数错误", "正方向 end 必须 > start")
                return
        if not 1 <= threshold <= 25:
            messagebox.showerror("参数错误", "阈值 n 必须在 1-25 之间")
            return
        cp = self.var_cp.get().strip()
        if not os.path.isabs(cp):
            cp = os.path.join(BASE_DIR, cp)
        # 默认从设定起点重新开始：删除旧 checkpoint，避免 exe 自动续传
        if os.path.isfile(cp):
            try:
                os.remove(cp)
            except OSError:
                pass

        cmd = [EXE, "--start", str(start), "--threshold", str(threshold)]
        if neg:
            cmd += ["--neg"]
        if not self.var_pause.get():     # 取消勾选「找到一个就暂停」→ 找到继续
            cmd += ["--continue"]
        if end_text:
            cmd += ["--end", end_text]
        if cp:
            cmd += ["--checkpoint", cp]
        self.cont_mode = not self.var_pause.get()

        self.running = True
        self.start_time = time.time()
        self.checked = 0
        self._set_status("状态: 运行中")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        end_desc = end_text if end_text else "∞（无限）"
        self._append_output(
            f"[{time.strftime('%H:%M:%S')}] 开始搜索：从种子 {start:,} 向{('负' if neg else '正')}"
            f"方向到 {end_desc}，阈值 {threshold}\n")
        self._init_log([
            f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"操作: 开始搜索",
            f"start: {start}",
            f"end: {end_text if end_text else '无限'}",
            f"方向: {'负' if neg else '正'}",
            f"阈值 n: {threshold}",
            f"窗口: {WINDOW}x{WINDOW}  网格: {GRID}x{GRID}",
            f"GPU: {os.path.basename(EXE)}",
        ])
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as e:
            self.running = False
            self._set_status(f"状态: 启动失败 - {e}")
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            return
        threading.Thread(target=self._reader, daemon=True).start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        if getattr(self, "cont_mode", False):
            # 找到继续模式：收尾统一由 _finish_cont 处理（避免重复打印/重命名日志）
            return
        self._cleanup_proc()
        # 打印暂停位置（checkpoint 精度 10 万种子）
        cp = self.var_cp.get().strip()
        if not os.path.isabs(cp):
            cp = os.path.join(BASE_DIR, cp)
        try:
            with open(cp, "r") as f:
                paused_at = int(f.read().strip())
            self._append_output(
                f"[{time.strftime('%H:%M:%S')}] 暂停：已算到种子 {paused_at:,}。"
                f"要继续请把 start 改为 {paused_at:,} 再点开始\n")
            self._set_status(f"状态: 已停止（算到种子 {paused_at:,}）")
            self._append_log([
                f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"操作: 暂停",
                f"已算到种子: {paused_at}",
                f"继续方式: 把 start 改为 {paused_at} 再点开始",
            ])
            self._rename_log(f"暂停-{paused_at}")
        except (OSError, ValueError):
            self._append_output(f"[{time.strftime('%H:%M:%S')}] 暂停：进度未知（运行时间过短）\n")
            self._set_status("状态: 已停止（进度未知）")

    # ---------- 子进程输出解析（后台线程） ----------
    def _reader(self):
        if getattr(self, "cont_mode", False):
            # 找到继续模式：逐行实时打印每个 FOUND 组
            group = []
            n = 0
            try:
                for line in self.proc.stdout:
                    line = line.rstrip()
                    if line.startswith("=== FOUND dense seed ==="):
                        group = [line]
                    elif line == "---" and group:
                        text = "\n".join(group)
                        n += 1
                        self.root.after(0, self._append_output, text + "\n")
                        group = []
                    elif group:
                        group.append(line)
            except Exception:
                pass
            finally:
                self.proc.wait()
            self.root.after(0, self._finish_cont, n)
            return

        found = False
        buf = []
        try:
            for line in self.proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                if RE_FOUND.search(line) and not found:
                    found = True
                buf.append(line)
                if len(buf) > 40:  # 只保留最后 40 行，避免内存膨胀
                    buf = buf[-40:]
                if found and RE_COUNT.search(line):
                    break
        except Exception:
            pass
        finally:
            self.proc.wait()
            out = "\n".join(buf).strip()

        # 解析结果
        if found:
            # 只保留 FOUND 结果段（过滤 searched 进度行）
            idx = out.find("=== FOUND dense seed ===")
            if idx >= 0:
                out = out[idx:]
            text = out
        else:
            m_no = RE_NOFOUND.search(out)
            m_thr = RE_THROUGHPUT.search(out)
            text = ""
        self.root.after(0, self._finish, found, text)

    def _finish_cont(self, found_count: int):
        """找到继续模式结束：进程被停或到达终点"""
        self._cleanup_proc()
        cp = self.var_cp.get().strip()
        if not os.path.isabs(cp):
            cp = os.path.join(BASE_DIR, cp)
        try:
            with open(cp, "r") as f:
                checked = int(f.read().strip())
            self.checked = checked
        except (OSError, ValueError):
            checked = self.checked
        if found_count > 0:
            self._append_output(f"[{time.strftime('%H:%M:%S')}] 本轮共找到 {found_count} 个种子（已搜索到 {checked:,}）\n")
            self._set_status(f"状态: 停止 - 已找到 {found_count} 个，算到 {checked:,}")
            self._append_log([
                f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"操作: 结束（找到继续模式）",
                f"找到种子数: {found_count}",
                f"已搜索到种子: {checked}",
            ])
            self._rename_log(f"找到-{found_count}个")
        else:
            self._append_output(f"[{time.strftime('%H:%M:%S')}] 本轮未找到（已搜索到 {checked:,}）\n")
            self._set_status(f"状态: 停止 - 未找到，算到 {checked:,}")
            self._append_log([
                f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"操作: 结束（找到继续模式）",
                f"找到种子数: 0",
                f"已搜索到种子: {checked}",
            ])
            self._rename_log("未找到")

    def _finish(self, found: bool, text: str):
        self._cleanup_proc()
        if found:
            self._append_output(f"[{time.strftime('%H:%M:%S')}] === 找到密集种子 ===\n" + text + "\n")
            self._set_status("状态: 完成 - 找到种子 ✅")
            m_seed = RE_FOUND.search(text)
            seed_no = m_seed.group(1) if m_seed else "?"
            self._append_log([
                f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"操作: 完成 - 找到",
                f"结果:\n{text}",
            ])
            self._rename_log(f"找到-{seed_no}")
        else:
            # 未找到：只追加一条总结（搜索过程中不刷屏）
            cp = self.var_cp.get().strip()
            try:
                with open(cp, "r") as f:
                    checked = int(f.read().strip())
                self.checked = checked
            except (OSError, ValueError):
                checked = self.checked
            self._append_output(f"[{time.strftime('%H:%M:%S')}] 完成：未找到（已搜索 {checked:,} 种子）\n")
            self._set_status(f"状态: 完成 - 未找到（已搜索 {checked:,} 种子）")
            self._append_log([
                f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"操作: 完成 - 未找到",
                f"已搜索种子数: {checked}",
            ])
            self._rename_log("未找到")

    def _cleanup_proc(self):
        if self.proc:
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        self.running = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

    # ---------- 进度轮询（主线程，1s 一次） ----------
    def _poll(self):
        if self.running and self.proc and self.proc.poll() is None:
            cp = self.var_cp.get().strip()
            if os.path.isfile(cp):
                try:
                    with open(cp, "r") as f:
                        self.checked = int(f.read().strip())
                except (OSError, ValueError):
                    pass
            try:
                start = int(self.var_start.get())
            except ValueError:
                start = 0
            done = abs(self.checked - start)   # 正负方向统一取绝对值
            elapsed = time.time() - self.start_time
            rate = done / elapsed if elapsed > 0 else 0
            self.var_progress.set(
                f"进度: 已搜索 {done:,} 种子 | 平均 {rate/1e6:.2f}M seeds/s | 耗时 {elapsed:.0f}s")
        self.root.after(1000, self._poll)

    # ---------- UI 辅助 ----------
    def _set_status(self, s: str):
        self.var_status.set(s)

    def _append_output(self, s: str):
        self.txt.config(state="normal")
        self.txt.insert("end", s)
        self.txt.see("end")
        self.txt.config(state="disabled")

    def _clear_output(self):
        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.config(state="disabled")

    def on_close(self):
        if self.running and self.proc and self.proc.poll() is None:
            if messagebox.askyesno("确认", "搜索仍在运行，确定要终止并退出吗？"):
                self.proc.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
