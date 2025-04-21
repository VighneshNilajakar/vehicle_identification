from plyer import notification

def notify_user(message):
    notification.notify(
        title="Vehicle Info",
        message=message,
        timeout=5
    )