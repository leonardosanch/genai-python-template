---
name: Email Transactional
description: Patterns for async transactional email with fastapi-mail, templates, and provider abstraction.
---

# Skill: Email Transactional

## Description

This skill covers production-ready transactional email for Python backend applications. Use this when implementing password reset flows, notification emails, confirmation emails, interview scheduling notifications, or any automated email sending.

## Executive Summary

**Critical email rules:**
- ALWAYS send emails asynchronously — never block the request/response cycle for email delivery
- Abstract email providers behind a port (`EmailSenderPort`) — infrastructure adapter handles SMTP, SendGrid, or AWS SES
- NEVER hardcode SMTP credentials — use `pydantic-settings` with environment variables exclusively
- NEVER expose sensitive tokens or internal URLs in email bodies — use short-lived, signed tokens
- Use Jinja2 templates for all HTML emails — never build HTML strings in Python code
- Celery tasks for bulk emails or emails with attachments — keep API response times fast

**Read full skill when:** Implementing password reset, sending notification emails, configuring SMTP providers, creating email templates, or testing email functionality.

---

## Versiones y Advertencias de Dependencias

| Dependencia | Versión Mínima | Estabilidad |
|-------------|----------------|-------------|
| fastapi-mail | >= 1.4.0 | ✅ Estable |
| jinja2 | >= 3.1.0 | ✅ Estable |
| aiosmtplib | >= 3.0.0 | ✅ Estable (usado internamente por fastapi-mail) |
| python-jose | >= 3.3.0 | ✅ Estable (para tokens en emails) |

> ⚠️ **fastapi-mail**: A partir de v1.4.0, usa `ConnectionConfig` en lugar de `ConnectionSettings`. Verificar docs si se usa versión anterior.

---

## Deep Dive

## Core Concepts

1. **Transactional Email**: Emails triggered by user actions (password reset, confirmation, notification). Not marketing/bulk email.

2. **Async Delivery**: Email sending MUST be non-blocking. Use `await` with async SMTP clients or dispatch to Celery tasks. A slow SMTP server should never delay API responses.

3. **Provider Abstraction**: Define an `EmailSenderPort` in the domain layer. Infrastructure adapters implement it for SMTP, SendGrid API, AWS SES, etc. Swapping providers requires zero domain/application changes.

4. **Template Engine**: Jinja2 templates for all HTML emails. Templates live in a dedicated directory, are versioned with the codebase, and support i18n.

5. **Token Security**: Password reset and confirmation tokens are short-lived (15-30 min), single-use, and signed (HMAC or JWT). Never use sequential or guessable tokens.

---

## External Resources

### 📧 Libraries & Frameworks

#### Email Libraries
- **fastapi-mail**: [github.com/sabuhish/fastapi-mail](https://github.com/sabuhish/fastapi-mail)
    - *Best for*: Async email sending with FastAPI, Jinja2 templates, attachments
- **aiosmtplib**: [aiosmtplib.readthedocs.io](https://aiosmtplib.readthedocs.io/)
    - *Best for*: Low-level async SMTP client (used by fastapi-mail internally)

#### Email Providers
- **SendGrid Python SDK**: [github.com/sendgrid/sendgrid-python](https://github.com/sendgrid/sendgrid-python)
    - *Best for*: Transactional email via API (no SMTP needed), delivery analytics
- **AWS SES with aiobotocore**: [docs.aws.amazon.com/ses/](https://docs.aws.amazon.com/ses/latest/dg/Welcome.html)
    - *Best for*: Serverless email at scale on AWS
- **Resend**: [resend.com/docs](https://resend.com/docs)
    - *Best for*: Modern developer-first email API with React email templates

#### Templates
- **Jinja2 Documentation**: [jinja.palletsprojects.com](https://jinja.palletsprojects.com/)
    - *Best for*: Template engine for HTML emails, inheritance, filters
- **MJML**: [mjml.io](https://mjml.io/)
    - *Best for*: Responsive email markup language (compile to HTML)
- **Maizzle**: [maizzle.com](https://maizzle.com/)
    - *Best for*: Tailwind CSS for email templates

---

### 🛡️ Email Security & Deliverability

- **SPF, DKIM, DMARC Guide**: [dmarcian.com/start-dmarc/](https://dmarcian.com/start-dmarc/)
    - *Best for*: Email authentication to prevent spam/phishing flags
- **OWASP Forgot Password Cheatsheet**: [cheatsheetseries.owasp.org/cheatsheets/Forgot_Password_Cheatsheet.html](https://cheatsheetseries.owasp.org/cheatsheets/Forgot_Password_Cheatsheet.html)
    - *Best for*: Secure password reset flows
- **Can I Email**: [caniemail.com](https://www.caniemail.com/)
    - *Best for*: Email client CSS/HTML compatibility (like caniuse for email)

---

### 🧪 Testing

- **MailHog**: [github.com/mailhog/MailHog](https://github.com/mailhog/MailHog)
    - *Best for*: Local SMTP testing server with web UI (Docker: `mailhog/mailhog`)
- **Mailtrap**: [mailtrap.io](https://mailtrap.io/)
    - *Best for*: Email testing sandbox for staging environments
- **Litmus**: [litmus.com](https://www.litmus.com/)
    - *Best for*: Email rendering testing across clients

---

## Decision Trees

### Decision Tree 1: Qué método de envío usar

```
¿Qué tipo de email necesitas enviar?
├── Transaccional simple (password reset, confirmación)
│   └── ¿Volumen < 100 emails/día?
│       ├── SÍ → fastapi-mail con SMTP directo (Gmail, Outlook, SMTP corporativo)
│       └── NO → SendGrid API o AWS SES (deliverability, analytics)
├── Notificaciones (alertas, updates, entrevistas)
│   └── ¿Necesitas tracking (open rate, clicks)?
│       ├── SÍ → SendGrid API (webhooks de eventos)
│       └── NO → fastapi-mail en Celery task (async, retry)
├── Bulk / Marketing
│   └── FUERA DEL ALCANCE — usar Mailchimp, SendGrid Marketing, etc.
└── Email con attachments (CVs, reportes)
    └── Celery task + fastapi-mail (nunca en el request/response cycle)
```

### Decision Tree 2: Sync vs Async vs Background

```
¿Cuándo enviar el email?
├── El usuario DEBE saber inmediatamente si se envió
│   └── await send_email() en el endpoint (async, pero en el request)
│       └── Timeout: max 10 segundos, fallback a retry queue
├── El usuario NO necesita confirmación inmediata
│   └── Celery task (fire-and-forget)
│       ├── Retry: 3 intentos con exponential backoff
│       └── Dead letter queue para emails fallidos
├── Envío masivo (> 50 destinatarios)
│   └── Celery task + batch processing
│       ├── Rate limit: respetar límites del provider
│       └── Nunca enviar 1000 emails en un loop síncrono
└── Envío programado (futuro)
    └── Celery Beat + task
```

---

## Instructions for the Agent

1.  **Provider Abstraction**: Define `EmailSenderPort` en domain layer. Nunca importar `fastapi_mail` fuera de infrastructure.

2.  **Credenciales**: SIEMPRE vía `pydantic-settings`. Variables requeridas: `MAIL_USERNAME`, `MAIL_PASSWORD`, `MAIL_FROM`, `MAIL_SERVER`, `MAIL_PORT`, `MAIL_STARTTLS`, `MAIL_SSL_TLS`.

3.  **Templates**: Todas las plantillas HTML en `src/infrastructure/email/templates/`. Usar herencia de Jinja2 (`{% extends "base.html" %}`). Nunca construir HTML con f-strings.

4.  **Tokens en emails**: Usar JWT con `exp` corto (15-30 min) y `jti` (token ID único). Marcar como usado en DB después del primer uso. Nunca usar UUIDs simples sin firma.

5.  **Error handling**: Capturar `SMTPException`, `ConnectionError`, `TimeoutError`. Log del error con `structlog`. Nunca exponer errores SMTP al usuario — responder siempre con mensaje genérico ("Si el email existe, recibirás instrucciones").

6.  **Testing**: Usar `unittest.mock.AsyncMock` para mockear el sender en unit tests. Para integration tests, usar MailHog (Docker).

7.  **Seguridad de reset password**: NUNCA revelar si un email existe o no. Siempre responder con el mismo mensaje y mismo HTTP status.

---

## Code Examples

### Example 1: Port en Domain Layer

```python
# src/domain/ports/email_sender_port.py
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EmailMessage:
    """Domain value object for email messages."""

    to: list[str]
    subject: str
    template_name: str
    template_data: dict[str, str]
    attachments: list[str] | None = None


class EmailSenderPort(ABC):
    """Domain port — infrastructure implements this."""

    @abstractmethod
    async def send(self, message: EmailMessage) -> None:
        """Send a single email. Raises EmailDeliveryError on failure."""

    @abstractmethod
    async def send_bulk(self, messages: list[EmailMessage]) -> None:
        """Send multiple emails. Dispatches to background tasks."""
```

### Example 2: Infrastructure Adapter con fastapi-mail

```python
# src/infrastructure/email/smtp_email_sender.py
import structlog
from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

from src.domain.ports.email_sender_port import EmailMessage, EmailSenderPort
from src.domain.exceptions import EmailDeliveryError

logger = structlog.get_logger()

TEMPLATE_DIR = Path(__file__).parent / "templates"


class SmtpEmailSender(EmailSenderPort):
    """SMTP adapter using fastapi-mail."""

    def __init__(
        self,
        mail_server: str,
        mail_port: int,
        mail_username: str,
        mail_password: str,
        mail_from: str,
        mail_starttls: bool = True,
        mail_ssl_tls: bool = False,
    ) -> None:
        self._config = ConnectionConfig(
            MAIL_USERNAME=mail_username,
            MAIL_PASSWORD=mail_password,
            MAIL_FROM=mail_from,
            MAIL_PORT=mail_port,
            MAIL_SERVER=mail_server,
            MAIL_STARTTLS=mail_starttls,
            MAIL_SSL_TLS=mail_ssl_tls,
            TEMPLATE_FOLDER=TEMPLATE_DIR,
        )
        self._mailer = FastMail(self._config)
        self._jinja_env = Environment(
            loader=FileSystemLoader(TEMPLATE_DIR),
            autoescape=select_autoescape(["html", "xml"]),
        )

    async def send(self, message: EmailMessage) -> None:
        """Send a single transactional email."""
        try:
            msg = MessageSchema(
                subject=message.subject,
                recipients=message.to,
                template_body=message.template_data,
                subtype=MessageType.html,
            )
            await self._mailer.send_message(msg, template_name=message.template_name)
            logger.info(
                "email_sent",
                to=message.to,
                subject=message.subject,
                template=message.template_name,
            )
        except Exception as exc:
            logger.error(
                "email_send_failed",
                to=message.to,
                error=str(exc),
            )
            raise EmailDeliveryError(f"Failed to send email: {exc}") from exc

    async def send_bulk(self, messages: list[EmailMessage]) -> None:
        """Send bulk emails — should be called from Celery task."""
        for message in messages:
            await self.send(message)
```

### Example 3: Use Case — Password Reset

```python
# src/application/use_cases/request_password_reset.py
import structlog
from datetime import timedelta

from src.domain.ports.email_sender_port import EmailMessage, EmailSenderPort
from src.domain.ports.user_repository_port import UserRepositoryPort
from src.infrastructure.security.jwt_handler import JWTHandler

logger = structlog.get_logger()

RESET_TOKEN_EXPIRY = timedelta(minutes=30)


class RequestPasswordResetUseCase:
    """Sends a password reset email if the user exists.

    SECURITY: Always returns success, even if email not found.
    Never reveal whether an email exists in the system.
    """

    def __init__(
        self,
        user_repo: UserRepositoryPort,
        email_sender: EmailSenderPort,
        jwt_handler: JWTHandler,
        frontend_url: str,
    ) -> None:
        self._user_repo = user_repo
        self._email_sender = email_sender
        self._jwt = jwt_handler
        self._frontend_url = frontend_url

    async def execute(self, email: str) -> None:
        """Request password reset. Always succeeds from user perspective."""
        user = await self._user_repo.find_by_email(email)

        if user is None:
            # SECURITY: Do NOT reveal that email doesn't exist
            logger.info("password_reset_requested_unknown_email", email_hash=hash(email))
            return

        # Generate short-lived, signed token
        token = self._jwt.create_token(
            subject=str(user.id),
            token_type="password_reset",
            expires_delta=RESET_TOKEN_EXPIRY,
        )

        reset_url = f"{self._frontend_url}/reset-password?token={token}"

        await self._email_sender.send(
            EmailMessage(
                to=[user.email],
                subject="Recuperación de contraseña",
                template_name="password_reset.html",
                template_data={
                    "user_name": user.full_name,
                    "reset_url": reset_url,
                    "expiry_minutes": "30",
                },
            )
        )
        logger.info("password_reset_email_sent", user_id=str(user.id))
```

### Example 4: Jinja2 Template

```html
<!-- src/infrastructure/email/templates/base.html -->
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; background: #f4f4f7; margin: 0; padding: 0; }
        .container { max-width: 600px; margin: 0 auto; background: #ffffff; }
        .header { background: #2563eb; padding: 24px; text-align: center; }
        .header h1 { color: #ffffff; margin: 0; font-size: 20px; }
        .content { padding: 32px 24px; color: #333; line-height: 1.6; }
        .btn { display: inline-block; background: #2563eb; color: #fff; padding: 12px 32px;
               border-radius: 6px; text-decoration: none; font-weight: bold; margin: 16px 0; }
        .footer { padding: 16px 24px; text-align: center; color: #999; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header"><h1>{% block header %}TalentFinder{% endblock %}</h1></div>
        <div class="content">{% block content %}{% endblock %}</div>
        <div class="footer">
            {% block footer %}
            <p>Este es un email automático, no responder.</p>
            <p>&copy; {{ current_year }} TalentFinder. Todos los derechos reservados.</p>
            {% endblock %}
        </div>
    </div>
</body>
</html>
```

```html
<!-- src/infrastructure/email/templates/password_reset.html -->
{% extends "base.html" %}

{% block content %}
<p>Hola {{ user_name }},</p>
<p>Recibimos una solicitud para restablecer tu contraseña.</p>
<p>Haz clic en el siguiente botón para crear una nueva contraseña:</p>
<p style="text-align: center;">
    <a href="{{ reset_url }}" class="btn">Restablecer Contraseña</a>
</p>
<p>Este enlace expira en <strong>{{ expiry_minutes }} minutos</strong>.</p>
<p>Si no solicitaste este cambio, puedes ignorar este email. Tu contraseña no será modificada.</p>
{% endblock %}
```

### Example 5: Celery Task para Email Async

```python
# src/infrastructure/tasks/email_tasks.py
from celery import shared_task
import structlog

logger = structlog.get_logger()


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,  # 60 seconds between retries
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,  # Exponential backoff: 60s, 120s, 240s
    retry_backoff_max=600,  # Max 10 minutes
)
def send_email_task(self, to: list[str], subject: str, template_name: str, template_data: dict) -> None:
    """Background email task with automatic retry.

    Usage:
        send_email_task.delay(
            to=["user@example.com"],
            subject="Entrevista programada",
            template_name="interview_scheduled.html",
            template_data={"candidate_name": "John", "date": "2025-02-10"},
        )
    """
    import asyncio
    from src.infrastructure.container import Container

    async def _send() -> None:
        container = Container()
        email_sender = container.email_sender()
        from src.domain.ports.email_sender_port import EmailMessage

        await email_sender.send(
            EmailMessage(
                to=to,
                subject=subject,
                template_name=template_name,
                template_data=template_data,
            )
        )

    asyncio.run(_send())
    logger.info("email_task_completed", to=to, subject=subject)
```

### Example 6: Settings con pydantic-settings

```python
# src/infrastructure/config/email_settings.py
from pydantic_settings import BaseSettings


class EmailSettings(BaseSettings):
    """Email configuration — all values from environment variables."""

    MAIL_USERNAME: str
    MAIL_PASSWORD: str  # App password or API key, never raw password
    MAIL_FROM: str
    MAIL_FROM_NAME: str = "TalentFinder"
    MAIL_SERVER: str = "smtp.gmail.com"
    MAIL_PORT: int = 587
    MAIL_STARTTLS: bool = True
    MAIL_SSL_TLS: bool = False

    # For development: disable actual sending
    MAIL_SUPPRESS_SEND: bool = False

    model_config = {"env_prefix": "", "case_sensitive": True}
```

---

## Anti-Patterns to Avoid

### ❌ Synchronous Email in Request Cycle
**Problem**: Blocking the API response while waiting for SMTP
**Example**:
```python
# BAD: Synchronous email blocks the request
@app.post("/register")
def register(user: UserCreate):
    db_user = create_user(user)
    smtplib.SMTP("smtp.gmail.com").send_message(msg)  # BLOCKS 2-5 seconds
    return db_user
```
**Solution**: Async sending or Celery task
```python
# GOOD: Non-blocking email
@app.post("/register")
async def register(user: UserCreate):
    db_user = await create_user(user)
    send_email_task.delay(to=[user.email], ...)  # Returns immediately
    return db_user
```

### ❌ Revealing Email Existence
**Problem**: Different responses for existing vs non-existing emails leak user data
**Example**:
```python
# BAD: Reveals whether email exists
@app.post("/forgot-password")
async def forgot_password(email: str):
    user = await user_repo.find_by_email(email)
    if not user:
        raise HTTPException(404, "Email no encontrado")  # INFORMATION LEAK
```
**Solution**: Always return the same response
```python
# GOOD: Same response regardless
@app.post("/forgot-password")
async def forgot_password(email: str):
    await reset_use_case.execute(email)  # Silently skips if not found
    return {"message": "Si el email existe, recibirás instrucciones"}
```

### ❌ Building HTML in Python Code
**Problem**: Unmaintainable, no escaping, XSS risk
**Example**:
```python
# BAD: HTML as f-string
body = f"<h1>Hola {user_name}</h1><p>Tu token es {token}</p>"
```
**Solution**: Jinja2 templates with autoescape
```python
# GOOD: Template with auto-escaping
template = jinja_env.get_template("password_reset.html")
body = template.render(user_name=user_name, reset_url=reset_url)
```

### ❌ Long-Lived Reset Tokens
**Problem**: Tokens valid for days create a security window
**Solution**: Max 30 minutes expiry. Single-use (mark as consumed in DB). Signed with HMAC/JWT.

### ❌ Hardcoded SMTP Credentials
**Problem**: Credentials in source code
**Solution**: `pydantic-settings` + environment variables. `.env` in `.gitignore`.

---

## Email Checklist

### Configuration
- [ ] SMTP credentials in environment variables (never in code)
- [ ] `pydantic-settings` for email configuration
- [ ] Connection tested in local development (MailHog or Mailtrap)
- [ ] `MAIL_SUPPRESS_SEND` flag for testing environments

### Templates
- [ ] Base template with consistent branding
- [ ] All templates use Jinja2 autoescape
- [ ] Templates tested for rendering correctness
- [ ] Responsive HTML (tested on Gmail, Outlook, Apple Mail)
- [ ] No user-controlled content rendered without escaping

### Security
- [ ] Reset tokens are JWT with short TTL (≤ 30 min)
- [ ] Tokens are single-use (consumed flag in DB)
- [ ] Forgot password never reveals email existence
- [ ] No sensitive data in email body (no raw passwords, no internal URLs)
- [ ] SPF, DKIM, DMARC configured on sending domain

### Reliability
- [ ] Async sending (non-blocking to API response)
- [ ] Celery tasks for bulk or attachment emails
- [ ] Retry with exponential backoff (3 attempts)
- [ ] Dead letter queue for permanently failed emails
- [ ] Structured logging for all email events (sent, failed, retried)

### Testing
- [ ] Unit tests mock EmailSenderPort
- [ ] Integration tests use MailHog (Docker)
- [ ] Template rendering tests verify variables
- [ ] Password reset flow tested end-to-end

---

## Additional References

- [FastAPI-Mail Tutorial](https://sabuhish.github.io/fastapi-mail/)
- [OWASP Forgot Password Cheatsheet](https://cheatsheetseries.owasp.org/cheatsheets/Forgot_Password_Cheatsheet.html)
- [Email HTML/CSS Compatibility — Can I Email](https://www.caniemail.com/)
- [SendGrid Python Quickstart](https://docs.sendgrid.com/for-developers/sending-email/quickstart-python)
- [Jinja2 Template Designer Documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/)
- [MailHog Docker Setup](https://github.com/mailhog/MailHog#docker)
