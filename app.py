from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

import traceback
import uvicorn
import nest_asyncio

from pydantic import BaseModel, Field

from backend import run_research_agent, resume_research_agent


load_dotenv()

nest_asyncio.apply()


app = FastAPI()


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request Models
# --------------------------------------------------

class ResearchRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = ""


# --------------------------------------------------
# Research API
# --------------------------------------------------

@app.post("/api/research")
def research_agent(request_data: ResearchRequest):

    try:

        print("1. Research request received")

        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty."
                }
            )

        print("2. User query:", user_message)

        thread_id = request_data.thread_id

        if thread_id is None:
            print("No thread_id provided. New research thread will be created.")
        else:
            print("Existing thread_id:", thread_id)

        # ------------------------------------------
        # Run Research Agent
        # ------------------------------------------

        result = run_research_agent(
            user_input=user_message,
            thread_id=thread_id
        )

        print("3. Research completed")

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as e:

        print("RESEARCH ERROR:", e)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


# --------------------------------------------------
# Human Approval / Revision
# --------------------------------------------------

@app.post("/api/research/approve")
async def approve_research(request_data: ApprovalRequest):

    try:

        # If user rejects the report,
        # feedback is required.

        if (
            not request_data.approved
            and not request_data.feedback.strip()
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": (
                        "Please provide revision feedback "
                        "when rejecting the research report."
                    ),
                },
            )

        print("Approval received")
        print("Thread ID:", request_data.thread_id)
        print("Approved:", request_data.approved)

        # ------------------------------------------
        # Resume LangGraph
        # ------------------------------------------

        result = resume_research_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except ValueError as exc:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(exc),
            },
        )

    except Exception as exc:

        print("APPROVAL ERROR:", exc)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/")
def start():

    return {
        "message": "Research Agent backend is running."
    }


# --------------------------------------------------
# Run Server
# --------------------------------------------------

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
