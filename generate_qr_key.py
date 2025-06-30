from cryptography.fernet import Fernet

# Generate a new key
key = Fernet.generate_key()

# Save the key to a file
with open("qr_key.key", "wb") as key_file:
    key_file.write(key)

print("✅ Encryption key generated and saved as 'qr_key.key'")
