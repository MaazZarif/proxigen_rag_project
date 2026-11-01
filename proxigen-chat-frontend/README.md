# Proxigen Chat Frontend

A lightweight, responsive, vanilla HTML/CSS/JavaScript chat widget. It calls the existing FastAPI `POST /chat` endpoint and is designed to be moved into WordPress later without requiring React or a frontend build system.

## 1. Run locally

1. Start your FastAPI backend, normally at `http://127.0.0.1:8000`.
2. Open `index.html` in a browser, or serve this folder through a local static server.
3. If your browser blocks the request, use a local server rather than opening the HTML as `file://`.
4. Configure FastAPI CORS for the exact frontend origin you use.

## 2. Backend contract

The widget sends:

```json
{
  "question": "What products do you offer?",
  "thread_id": "a-browser-session-conversation-uuid"
}
```

It expects a response like:

```json
{
  "thread_id": "a-browser-session-conversation-uuid",
  "question": "What products do you offer?",
  "answer": "..."
}
```

This matches the current FastAPI `ChatRequest` and `ChatResponse` schemas.

## 3. Configure the API URL

At the top of `chat-widget.js`, update:

```js
const API_BASE_URL = "http://127.0.0.1:8000";
```

For deployment, use the HTTPS URL of your hosted API, for example `https://api.example.com`. Do not leave localhost in a live WordPress site.

## 4. Conversation behavior

- The frontend creates a UUID for the browser tab's conversation and sends it as `thread_id`.
- It stores the thread ID and displayed messages in `sessionStorage`.
- Refreshing the page in the same tab keeps the displayed conversation.
- Closing the tab ends this session-storage persistence.
- **New conversation** clears the current tab's thread ID and message history.
- The backend's LangGraph checkpointer stores server-side graph state under the thread ID. The browser history is only for rendering messages.

This is guest chat, not authentication. A thread ID is not a user identity or an authorization mechanism.

## 5. WordPress integration later

For a simple initial integration, copy `chat-widget.css` and `chat-widget.js` into a small custom WordPress plugin or your child theme, enqueue them, and add this markup to the page:

```html
<div id="proxigen-chat-root"></div>
```

Enqueue the CSS and JavaScript through WordPress rather than pasting the entire implementation into a page editor. Update `API_BASE_URL` to the deployed backend URL and configure CORS to allow your WordPress origin.

If the widget should be available on every page, enqueue the assets site-wide. If it should only appear on one page, enqueue them only on that page.

## 6. FastAPI CORS

The browser will require CORS permission when the WordPress/site origin differs from the API origin. In FastAPI, configure `CORSMiddleware` with explicit allowed origins for local development and the production WordPress domain. Avoid allowing every origin in production unless you have a deliberate reason.

## 7. Notes

- No frontend framework or build step is required.
- User and assistant text is rendered with `textContent`, not interpreted as HTML.
- The widget does not render Markdown; answers appear as plain text.
- The browser must be able to reach the API over the network.
- For production, add rate limiting, request-size limits, monitoring, and appropriate backend protections.
