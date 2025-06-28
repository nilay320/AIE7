# 📅 Cal.com Bookings with Context7 MCP and Cursor

Welcome to this hands-on mini-project where you'll use **Context7 MCP Server** to integrate with **Cal.com** — no coding required! By the end, you'll be able to schedule bookings in Cal.com directly from **Cursor**, using intelligent agent communication.

## ✅ Prerequisites

Before we begin, make sure you've completed the following setup:

1. **Create a Cal.com Account**  
   Sign up at [https://cal.com](https://cal.com) and complete your profile setup.

2. **Generate a Cal.com API Key**  
   Go to your Cal.com dashboard → *Settings* → *API* → *Create API Key*.  
   Save this key securely — you'll need it soon.

3. **Create Sample Bookings**  
   Add a few test bookings in your Cal.com account so you can later interact with them through the MCP agent.

## 🔌 What We're Building

You'll connect the **Context7 MCP Server** to Cursor, allowing intelligent agents to:

- Access your Cal.com bookings.
- Create new bookings via agent commands — **without writing code**.
- Seamlessly interact with your scheduling data in a conversational, interactive way.

To connect to the MCP server in Cursor, use the following line of code:

```bash
{
  "mcpServers": {
    "context7": {
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```

For troubleshooting you might install it also manually:

```bash
npx -y @upstash/context7-mcp@latest
```

and replace in Cursor Settings with:

```json
{
  "mcpServers": {
    "context7-cloud": {
      "url": "https://mcp.context7.com/mcp",
      "description": "Context7 cloud MCP server",
      "timeout": 30000,
      "retries": 3
    },
    "context7-local": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"],
      "description": "Local MCP server (runs Cal.com docs you ingested)"
    }
  }
}
```

## 🚀 Let's Get Started

Once your environment is ready, you'll connect the Cal.com tool to the Context7 MCP server, and try creating a booking from Cursor by simply asking your agent.

Have fun exploring the power of AI-enhanced automation — and remember: no lines of code, just context!

---

# Cal.com Meeting Scheduler

This project provides a Python script to automatically schedule meetings using the Cal.com API v2.

## Features

- Schedule meetings using Cal.com API v2
- Automatically calculates next Monday at 2 PM EST
- Handles timezone conversion (EST to UTC)
- Environment variable configuration for API keys

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your Cal.com API key:**
   - Go to [Cal.com Settings > Security](https://app.cal.com/settings/security)
   - Generate a new API key
   - Create a `.env` file in the project root:
   ```
   CAL_API_KEY=cal_live_your_actual_api_key_here
   ```

3. **Run the scheduler:**
   ```bash
   python schedule_meeting.py
   ```

## Configuration

The script is currently configured to:
- Schedule a 30-minute meeting
- Set meeting for next Monday at 2:00 PM EST
- Use the username: `nilay-jhaveri-wfh2ph`
- Invite attendee: `nilay320@yahoo.com`

## API Documentation

This script uses Cal.com API v2. Key endpoints:
- **POST /v2/bookings** - Create a booking
- **Authentication**: Bearer token with API key
- **Required headers**: 
  - `Authorization: Bearer {API_KEY}`
  - `cal-api-version: 2024-08-13`
  - `Content-Type: application/json`

## Meeting Details

- **Duration**: 30 minutes
- **Time**: Next Monday at 2:00 PM EST (automatically converted to UTC)
- **Attendee**: Nilay (nilay320@yahoo.com)
- **Host**: nilay-jhaveri-wfh2ph

## Error Handling

The script includes error handling for:
- Missing API key
- API request failures
- Network issues
- Invalid responses

## Notes

- The script automatically finds the next Monday from the current date
- Times are converted from EST to UTC for the API
- All bookings are created with timezone awareness

---
