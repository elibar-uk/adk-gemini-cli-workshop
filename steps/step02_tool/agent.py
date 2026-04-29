import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent

# =========================================
# STEP 3: Uncomment the following line to import Google Search
# =========================================
# from google.adk.tools import google_search
# =========================================

def now(city: str = "local") -> dict:
    """Returns current date/time, with basic city timezone support."""
    city_normalized = city.strip().lower()
    timezone_by_city = {
        "milan": "Europe/Rome",
        "rome": "Europe/Rome",
        "paris": "Europe/Paris",
        "london": "Europe/London",
        "new york": "America/New_York",
    }
    timezone_name = timezone_by_city.get(city_normalized)
    if timezone_name:
        current_dt = datetime.datetime.now(ZoneInfo(timezone_name))
    else:
        timezone_name = "local"
        current_dt = datetime.datetime.now()

    return {
        "status": "success",
        "city": city,
        "timezone": timezone_name,
        "current_time": current_dt.strftime("%Y-%m-%d %H:%M:%S")
    }

root_agent = Agent(
    name="travel_tool",
    model="gemini-2.5-flash",
    instruction=(
        "You are a helpful travel assistant. "
        "For any question about current time/date/day, always call the `now` tool first. "
        "If the user mentions a city, pass it as the `city` argument."
    ),
    tools=[
        now,
        # =========================================
        # STEP 3: Uncomment to register Google Search
        # =========================================
        # google_search,
        # =========================================
    ],
)
