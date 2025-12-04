from flask import Flask
from flask_cors import CORS
from pathlib import Path


def create_app():
    app = Flask(__name__)
    CORS(app)

    # Configuration - use /tmp for uploads/jobs to avoid triggering Flask reload
    app.config["UPLOAD_FOLDER"] = Path("/tmp/aq-benchmark/uploads")
    app.config["JOBS_DIR"] = Path("/tmp/aq-benchmark/jobs")
    app.config["DATA_DIR"] = Path(__file__).parent.parent / "data"
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

    # Ensure directories exist
    app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
    app.config["JOBS_DIR"].mkdir(parents=True, exist_ok=True)
    app.config["DATA_DIR"].mkdir(parents=True, exist_ok=True)

    # Register blueprints
    from app.routes.uploads import bp as uploads_bp
    from app.routes.benchmarks import bp as benchmarks_bp

    app.register_blueprint(uploads_bp)
    app.register_blueprint(benchmarks_bp)

    # Initialize benchmark manager
    from app.services.benchmark_manager import BenchmarkManager

    app.benchmark_manager = BenchmarkManager(
        jobs_dir=app.config["JOBS_DIR"],
        uploads_dir=app.config["UPLOAD_FOLDER"],
        data_dir=app.config["DATA_DIR"],
    )

    return app
