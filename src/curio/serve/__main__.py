import os
import uvicorn

uvicorn.run("curio.serve.app:app", host=os.getenv("CURIO_HOST", "127.0.0.1"), port=int(os.getenv("CURIO_PORT", "8000")))
