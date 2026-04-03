import time

import struct
from multiprocessing import shared_memory

# ================= CONFIG =================
SHM_NAME = "ml_detection_shm"
STRUCT_FORMAT = "iii"   # detected, x_offset, y_offset
STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)
# =========================================

# -------- Create or Attach Shared Memory --------
try:
    shm = shared_memory.SharedMemory(
        name=SHM_NAME,
        create=True,
        size=STRUCT_SIZE
    )
    print("Shared memory CREATED")
except FileExistsError:
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    print("Shared memory ATTACHED")

# -------- Initialize with zeros --------
struct.pack_into(STRUCT_FORMAT, shm.buf, 0, 0, 0, 0)

print("Writing (0, 0, 0) to shared memory...")
print("Press Ctrl+C to stop")

try:
    while True:
        # Continuously write zeros
        struct.pack_into(STRUCT_FORMAT, shm.buf, 0, 0, 0, 0)
        time.sleep(0.1)  # 10 Hz update rate

except KeyboardInterrupt:
    print("\nStopping writer...")

finally:
    shm.close()
    # Do NOT unlink here if reader is still running
    print("Shared memory closed (not unlinked)")




