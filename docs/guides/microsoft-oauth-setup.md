# Microsoft OAuth Setup Guide

This guide explains how to set up Microsoft OAuth to enable OneDrive, Outlook Mail, and OneNote document sources in LLM Relay.

## Prerequisites

- A Microsoft account (personal or work/school)
- Access to Azure Portal (https://portal.azure.com)

## Step 1: Register an Azure AD Application

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Azure Active Directory** > **App registrations**
3. Click **New registration**
4. Fill in the details:
   - **Name**: `LLM Relay` (or your preferred name)
   - **Supported account types**: Choose based on your needs:
     - *Personal Microsoft accounts only* - for OneDrive Personal, Outlook.com
     - *Accounts in any organizational directory and personal Microsoft accounts* - for both personal and work/school accounts
   - **Redirect URI**: Select **Web** and enter your callback URL:
     - Development: `http://localhost:8081/api/oauth/microsoft/callback`
     - Production: `https://your-domain/api/oauth/microsoft/callback`
5. Click **Register**

## Step 2: Configure API Permissions

1. In your app registration, go to **API permissions**
2. Click **Add a permission** > **Microsoft Graph** > **Delegated permissions**
3. Add the following permissions based on which services you need:

### For OneDrive:
- `Files.Read` - Read user files
- `Files.Read.All` - Read all files user can access (for shared files)

### For Outlook Mail:
- `Mail.Read` - Read user mail

### For OneNote:
- `Notes.Read` - Read user OneNote notebooks

### Required for all:
- `offline_access` - Maintain access to data (required for refresh tokens)
- `User.Read` - Sign in and read user profile (auto-added)

4. Click **Grant admin consent** if you're an admin (optional, users can consent individually)

## Step 3: Create Client Secret

1. Go to **Certificates & secrets**
2. Click **New client secret**
3. Add a description (e.g., "LLM Relay Production")
4. Choose an expiration period
5. Click **Add**
6. **Important**: Copy the secret value immediately - it won't be shown again!

## Step 4: Configure LLM Relay

Add the following environment variables to your LLM Relay configuration:

```bash
MICROSOFT_CLIENT_ID=<Application (client) ID from Overview page>
MICROSOFT_CLIENT_SECRET=<Client secret value from Step 3>
```

### Docker Compose Example

```yaml
services:
  llm-relay:
    environment:
      - MICROSOFT_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
      - MICROSOFT_CLIENT_SECRET=your-client-secret-here
```

## Step 5: Connect Your Account

1. Open LLM Relay Admin UI
2. Go to **Data Sources** > **Document Stores**
3. Click **Add Store**
4. Select a Microsoft source (OneDrive, Outlook Mail, or OneNote)
5. Click **Connect Microsoft Account**
6. Sign in with your Microsoft account
7. Grant the requested permissions
8. Select your folder/notebook preferences
9. Save the document store

## Troubleshooting

### "AADSTS50011: The reply URL does not match"
- Ensure your redirect URI in Azure exactly matches your LLM Relay URL
- Check for trailing slashes and http vs https

### "Need admin approval"
- Your organization may require admin consent for certain permissions
- Contact your Azure AD admin or use a personal Microsoft account

### "Invalid client secret"
- Client secrets expire - create a new one if expired
- Ensure no extra whitespace when copying the secret

### Token refresh fails
- Ensure `offline_access` permission is granted
- Re-authenticate by connecting the account again

## Security Best Practices

1. **Use separate apps for dev/production** - Different redirect URIs and secrets
2. **Rotate secrets regularly** - Set calendar reminders before expiration
3. **Minimize permissions** - Only request permissions you actually need
4. **Use HTTPS in production** - Required for OAuth security

## Supported Account Types

| Account Type | OneDrive | Outlook | OneNote |
|--------------|----------|---------|---------|
| Personal (outlook.com, hotmail.com) | Yes | Yes | Yes |
| Work/School (Microsoft 365) | Yes | Yes | Yes |

## API Endpoints Used

LLM Relay uses the Microsoft Graph API v1.0:

- OneDrive: `/me/drive/root/children`, `/me/drive/items/{id}/content`
- Outlook: `/me/messages`, `/me/mailFolders`
- OneNote: `/me/onenote/notebooks`, `/me/onenote/pages`
