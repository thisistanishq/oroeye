# 🚀 Deploying OroEYE: Share the Magic

So, you've got this beast running locally. That's cute.
But **OroEYE** deserves to be seen by the world.

Here is the **fastest, easiest, and free-est** way to get this live on the internet.

---

## ⚡ Option 0: The "Lazy Mode" (One-Click Render Blueprint)
**This is the absolute easiest way.** I added a `render.yaml` file for you.

1.  Push code to GitHub.
2.  Go to [dashboard.render.com/blueprints](https://dashboard.render.com/blueprints).
3.  Click **New Blueprint Instance**.
4.  Select your repo.
5.  Render will read the `render.yaml` and **auto-fill everything**.
6.  It will just ask you for the `MONGO_URI`. Paste it.
7.  Click **Apply**. Done.

---

## 🔧 Option 1: The Manual Method (Render.com)
If Option 0 is too magical for you, here is how to do it by hand:

### 1. Push to GitHub
Make sure your code is on GitHub. If it's not, what are you even doing?
```bash
git add .
git commit -m "Ready for launch"
git push
```

### 2. Connect to Render
1.  Go to [dashboard.render.com](https://dashboard.render.com/)
2.  Click **New +** -> **Web Service**.
3.  Connect your GitHub repo.

### 3. Configure the Beast
Use these EXACT settings. Don't mess this up.

*   **Name:** `oroeye-app` (or whatever cool name you want)
*   **Region:** Closest to you.
*   **Branch:** `main`
*   **Runtime:** `Python 3`
*   **Build Command:** `pip install -r backend/requirements.txt`
*   **Start Command:** `gunicorn --chdir backend app:app`

### 4. Feed it Secrets (Environment Variables)
Scroll down to **"Advanced"** -> **"Environment Variables"**.
Add these (or the app crashes):

| Key | Value |
| :--- | :--- |
| `MONGO_URI` | Your MongoDB connection string (e.g., `mongodb+srv://...`) |
| `SECRET_KEY` | Smash your keyboard (e.g., `sadf879s7d8f7s9d8f`) |
| `TF_ENABLE_ONEDNN_OPTS` | `0` (Optional, keeps logs clean) |

### 5. Launch 🚀
Click **Create Web Service**.
Render will build it (might take 2-3 mins).
When it's done, you'll get a live URL (e.g., `oroeye.onrender.com`).
**Share it. Flex it.**

---

## 🏠 Option 2: The "Hacker House" Method (Local Network)
Want to show it to friends on your WiFi without deploying?

1.  Find your local IP address (e.g., `192.168.1.X`).
2.  Edit `backend/app.py`:
    ```python
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5001)  # Change host to 0.0.0.0
    ```
3.  Run the app:
    ```bash
    python app.py
    ```
4.  Tell your friends to open `http://YOUR-IP-ADDRESS:5001` on their phones.
    *   *Note: Camera/Mic might not work without HTTPS on mobile Chrome. Use Firefox or Deploy to Render for full features.*

---

## 🐳 Option 3: Docker (For the Pros)
If you know what Docker is, you probably don't need this guide.
But if you do need it, here is the magic invocation:

1.  **Build the Container:**
    ```bash
    docker build -t oroeye .
    ```

2.  **Run the Beast:**
    ```bash
    docker run -p 5001:5001 oroeye
    ```

3.  **Visit:** `http://localhost:5001`

---

**That's it.**
Go spread the revolution.
