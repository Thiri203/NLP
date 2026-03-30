# 🚀 n8n Automation with PostgreSQL & ngrok

This repository contains a Docker-based setup for **n8n**, a powerful workflow automation tool. It uses **PostgreSQL 16** for persistent storage and **ngrok** to provide a secure public tunnel for receiving external webhooks (e.g., Line, Facebook, Stripe).

## 📂 Project Structure

```text
.
├── docker-compose.yaml    # Container orchestration and service definitions
├── .env                   # Environment variables (Credentials & URLs)
├── .env.key               # Encryption key for environment variables
├── .gitignore             # Specifies intentionally untracked files to ignore
└── README.md              # Project documentation
```

-----

## 🛠 Prerequisites

1.  **Docker & Docker Compose**: Ensure you have Docker Desktop or Docker Engine installed.
2.  **ngrok**: Installed and authenticated on your local machine.

-----

## 🚀 Getting Started

### 1\. Configuration

Create a `.env` file in the root directory. Copy and paste the following, then update with your desired values:

```env
# Database Configuration
DB_USER=n8n_admin
DB_PASSWORD=your_secure_password
DB_NAME=n8n_db

# ngrok Configuration
# Get your URL by running: ngrok http 5678
NGROK_URL=https://your-unique-id.ngrok-free.app
```

### 2\. Launching Services

Run the following command to start both the database and n8n in the background:

```bash
docker-compose up -d
```

### 3\. Accessing n8n

  * **Local UI:** [http://localhost:5678](https://www.google.com/search?q=http://localhost:5678)
  * **Public Webhooks:** Use the `NGROK_URL` provided in your `.env` for external services to reach your workflows.

-----

## ⚠️ Important Notes

### Persistent Data

Upon the first run, Docker will automatically create two directories:

  * `./postgres_data`: Stores all your database records (Workflows, executions).
  * `./n8n_data`: Stores n8n internal configurations.
    **Do not delete these folders unless you want to reset your entire setup.**

### Updating ngrok URL

If you are using a free ngrok account, the URL changes every time you restart ngrok. When it changes:

1.  Update the `NGROK_URL` in your `.env` file.
2.  Restart the container to apply changes: `docker-compose up -d`.

-----

## 🛠 Troubleshooting

### "Password authentication failed"

If you see this error in the logs, it means the database was initialized with a different password previously. To reset:

```bash
docker-compose down -v
rm -rf ./postgres_data
docker-compose up -d
```

*(Warning: This will wipe all existing workflows and data.)*
