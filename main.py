import uvicorn
from src.core.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=s.api_host,
        port=s.api_port,
        reload=s.api_debug,
    )
