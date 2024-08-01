class Message:
    def __init__(self, tok_id, i):
        self.tok_id = tok_id  # The last verified token id
        self.i = i  # The index of the last verified token


def message_listener(message_bus, servers):
    while True:
        sender_id, message = message_bus.get()
        for server in servers:
            server.update_state(message)
