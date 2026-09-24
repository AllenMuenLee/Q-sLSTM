from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys
import platform
from contextlib import redirect_stdout, redirect_stderr

# ===============================
# Path Construction
# ===============================

class Tee:
	def __init__(self, *streams):
		self.streams = streams
	def write(self, data):
		for s in self.streams:
			s.write(data)
			s.flush()
	def flush(self):
		for s in self.streams:
			s.flush()

def build_result_path(args, experiment_root: Path):

	timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S_%f")
	run_name = f"RUN_{timestamp}"
	MODEL_EXTRA_PARAMS = {
		"qlstm": [],
		"qslstm": [],
		"qslstm_log": [],
		"lstm": [],
		"self_modulating_qfwp": [],
		"self_modulating_qfwp_only_new_params": [],
		"self_modulating_qfwp_only_old_params": [],
		"standard_qfwp": [],
	}

	if args.model not in MODEL_EXTRA_PARAMS:
		raise ValueError("args.model incompatible")

	parts = [
		f"DATASET_{args.dataset}",
		f"MODEL_{args.model}",
	]

	# model-specific parameters
	for prefix, attr in MODEL_EXTRA_PARAMS[args.model]:
		value = getattr(args, attr)
		parts.append(f"{prefix}_{value}")

	# common parameters
	parts += [
		f"HIDDEN_SIZE_{args.hidden_size}",
		f"QNN_DEPTH_{args.qnn_depth}",
		f"SEQ_LEN_{args.window_len}",
		f"SEED_{args.seed}",
	]

	run_path = Path(*parts)

	save_dir = Path(args.save_dir)

	# --- 安全檢查 ---

	if any(part == ".." for part in run_path.parts):
		raise ValueError("run_path 不可包含 '..'")

	if any(part == ".." for part in save_dir.parts):
		raise ValueError("--save_dir 不可包含 '..'")

	if run_path.is_absolute():
		raise ValueError("run_path 不可為絕對路徑")

	if save_dir.is_absolute():
		raise ValueError("--save_dir 必須使用相對路徑（例如 results），不可使用絕對路徑")
	
	result_path = (experiment_root / save_dir / run_path / run_name).resolve()

	# 確保沒有跳出封存包
	result_path.relative_to(experiment_root)
	result_path.mkdir(parents=True, exist_ok=True)


	return result_path
# ===============================
# Metadata Saving
# ===============================

def save_args_json(args, result_path: Path):
	with open(result_path / "args.json", "w", encoding="utf-8") as f:
		json.dump(vars(args), f, indent=4, sort_keys=True)


def save_git_revision(result_path: Path):
	try:
		commit = subprocess.check_output(
			["git", "rev-parse", "HEAD"],
			stderr=subprocess.DEVNULL
		).decode().strip()

		branch = subprocess.check_output(
			["git", "rev-parse", "--abbrev-ref", "HEAD"],
			stderr=subprocess.DEVNULL
		).decode().strip()

		with open(result_path / "git_revision.txt", "w") as f:
			f.write(f"branch: {branch}\n")
			f.write(f"commit: {commit}\n")

	except Exception:
		with open(result_path / "git_revision.txt", "w") as f:
			f.write("Git information not available.\n")


def save_environment_snapshot(result_path: Path):
	"""
	產出：
	- environment.yaml      (conda env export, 若可用)
	- conda_list.txt        (conda list)
	- requirements.txt      (pip freeze)
	- python_info.txt       (python -V + executable)
	"""
	result_path.mkdir(parents=True, exist_ok=True)

	# 1) Python info（永遠可用）
	with open(result_path / "python_info.txt", "w") as f:
		f.write(f"python_version: {sys.version}\n")
		f.write(f"python_executable: {sys.executable}\n")

	# 2) pip freeze（永遠建議存，因為 conda export 可能漏 pip-only）
	try:
		with open(result_path / "requirements.txt", "w") as f:
			subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f, check=True)
	except Exception as e:
		with open(result_path / "pip_freeze_error.txt", "w") as f:
			f.write(repr(e) + "\n")

	# 3) conda 相關（如果 conda 存在才做）
	#    注意：你可能在非 conda 環境跑，這段會自動略過
	try:
		# 有些系統 conda 不在 PATH，但你在 conda env 裡通常會在
		subprocess.run(["conda", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

		# conda env export（含 pip: 區塊；但仍建議另存 requirements.txt）
		with open(result_path / "environment.yaml", "w") as f:
			subprocess.run(["conda", "env", "export"], stdout=f, stderr=subprocess.DEVNULL, check=True)

		# conda list（可快速 diff）
		with open(result_path / "conda_list.txt", "w") as f:
			subprocess.run(["conda", "list"], stdout=f, stderr=subprocess.DEVNULL, check=True)

		# 4) 更硬核：explicit spec（可 100% 還原同平台）
		#    注意：這個更偏「鎖死到平台」，跨平台不一定能用
		with open(result_path / "conda_explicit.txt", "w") as f:
			subprocess.run(["conda", "list", "--explicit"], stdout=f, stderr=subprocess.DEVNULL, check=True)

		# conda env name（方便記錄）
		env_name = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
		with open(result_path / "conda_env_name.txt", "w") as f:
			f.write(env_name + "\n")

	except Exception:
		# 不在 conda 或 conda 不可用 → 沒事
		pass


def model_config_block(args) -> str:
	return f"""
	## QLSTM Configuration

	### Architecture
	- Hidden size: {args.hidden_size}
	- QNN depth: {args.qnn_depth}
	- Sequence length: {args.window_len}
	"""

def generate_experiment_readme(args, result_path: Path) -> None:

	result_path = Path(result_path)
	readme_path = result_path / "README.md"

	command_line = " ".join(sys.argv)

	try:
		git_commit = subprocess.check_output(
			["git", "rev-parse", "HEAD"],
			stderr=subprocess.DEVNULL
		).decode().strip()
	except Exception:
		git_commit = "N/A"

	try:
		git_branch = subprocess.check_output(
			["git", "rev-parse", "--abbrev-ref", "HEAD"],
			stderr=subprocess.DEVNULL
		).decode().strip()
	except Exception:
		git_branch = "N/A"

	args_dict = vars(args)

	model_block = model_config_block(args)

	content = f"""
	# Experiment Run

	## Basic Info
	- Timestamp: {datetime.now().isoformat(timespec="seconds")}
	- Dataset: {getattr(args, "dataset", "N/A")}
	- Model: {getattr(args, "model", "N/A")}
	- Seed: {getattr(args, "seed", "N/A")}

	{model_block}
	
	---

	## Command Used

		{command_line}

	---

	## Git Info
	- Branch: {git_branch}
	- Commit: {git_commit}

	---

	## Full Arguments

		{json.dumps(args_dict, indent=4)}

	---

	## System Info
	- Python Version: {sys.version}
	- Python Executable: {sys.executable}
	- Platform: {platform.platform()}

	---

	## Notes
	This folder is a self-contained experiment artifact.
	To reproduce:
	1. Checkout the git commit above.
	2. Restore the environment.
	3. Run the command listed above.
	"""

	readme_path.write_text(content.strip() + "\n", encoding="utf-8")
