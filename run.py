import traceback
import sys

try:
    from main import app
    print("main.py imported OK")
except Exception as e:
    print("ERROR importing main.py:")
    traceback.print_exc()
    sys.exit(1)

try:
    import uvicorn
    print("uvicorn imported OK")
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
except Exception as e:
    print("ERROR starting uvicorn:")
    traceback.print_exc()