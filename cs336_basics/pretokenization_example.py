import os
from typing import BinaryIO
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

import multiprocessing as mp
from multiprocessing import Pool

def process_chunk_worker(chunk_data):
    """Worker function to process a single chunk."""
    chunk_text, chunk_id = chunk_data
    
    # Run pre-tokenization on your chunk and store the counts for each pre-token
    matches = re.finditer(PAT, chunk_text)
    token_counts = {}
    for match in matches:
        token = match.group(0)
        token_counts[token] = token_counts.get(token, 0) + 1
    
    print(f"Chunk {chunk_id}: {len(token_counts)} unique tokens")
    return token_counts

def process_chunk_worker_shared(chunk_data):
    """Worker function that updates a shared global counter."""
    chunk_text, chunk_id, global_token_counts = chunk_data
    
    # Run pre-tokenization on your chunk and update global counts
    matches = re.finditer(PAT, chunk_text)
    chunk_token_counts = {}
    for match in matches:
        token = match.group(0)
        chunk_token_counts[token] = chunk_token_counts.get(token, 0) + 1
    
    # Update global counter atomically
    for token, count in chunk_token_counts.items():
        if token in global_token_counts:
            global_token_counts[token] += count
        else:
            global_token_counts[token] = count
    
    print(f"Chunk {chunk_id}: {len(chunk_token_counts)} unique tokens (global total: {len(global_token_counts)})")

def pre_tokenize_parallel(file: BinaryIO, num_processes: int, split_special_token: bytes):
    """Process chunks in parallel using multiprocessing with shared global counter."""
    boundaries = find_chunk_boundaries(file, num_processes, split_special_token)
    
    # Create a manager for shared state
    with mp.Manager() as manager:
        # Create a shared dictionary for global token counts
        global_token_counts = manager.dict()
        
        # Read all chunks into memory first (since file objects can't be pickled)
        chunks_data = []
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            chunks_data.append((chunk, i, global_token_counts))
        
        # Process chunks in parallel
        with Pool(processes=num_processes) as pool:
            pool.map(process_chunk_worker_shared, chunks_data)
        
        # Convert shared dict to regular dict
        return dict(global_token_counts)
    
## return a dict of token counts str -> int
def pre_tokenize_parallel_util(file_path: str, num_processes: int, split_special_token: bytes) -> dict [str, int]:
    with open(file_path, "rb") as f:
        return pre_tokenize_parallel(f, num_processes, split_special_token)

if __name__ == "__main__":
    # Set multiprocessing start method (important for macOS)
    mp.set_start_method('spawn', force=True)
    
    file_path = "/Users/hangxinc/Desktop/CS336/assignment1-basics/data/data/TinyStoriesV2-GPT4-valid.txt"
    num_chunks = 10
    num_processes = min(mp.cpu_count(), num_chunks)  # Use available CPU cores
    
    print(f"Using {num_processes} processes to process {num_chunks} chunks")
    
    # Serial version (original) with global token_counts
    print("\n=== Serial Processing with Global Token Counts ===")
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_chunks, "<|endoftext|>".encode("utf-8"))
            
        # Global token counts that accumulate across all chunks
        token_counts = {}
        
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            matches = re.finditer(PAT, chunk)
            for match in matches:
                token = match.group(0)
                print(type(token))
                token_counts[token] = token_counts.get(token, 0) + 1
        print(f"\nSerial total unique tokens: {len(token_counts)}")
        print("\nSample tokens and counts:")
        for i, (token, count) in enumerate(sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]):
            print(f"  '{token}': {count}")
            if i >= 9:  # Show only top 10
                break
    # Parallel version with shared global counter
    print("\n=== Parallel Processing with Shared Global Counter ===")
    with open(file_path, "rb") as f:
        global_token_counts = pre_tokenize_parallel(f, num_processes, "<|endoftext|>".encode("utf-8"))
        
        print(f"\nTotal unique tokens across all chunks: {len(global_token_counts)}")
        
        # Show some sample tokens and their counts
        print("\nSample tokens and counts:")
        for i, (token, count) in enumerate(sorted(global_token_counts.items(), key=lambda x: x[1], reverse=True)[:10]):
            print(f"  '{token}': {count}")
            if i >= 9:  # Show only top 10
                break