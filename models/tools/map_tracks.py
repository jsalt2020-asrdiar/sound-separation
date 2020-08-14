import numpy as np

def mag_mse_loss(sep, ref):
  """Match one of the separated tracks to a single reference based on abs fft distance.
  
  Args:
    sep: (Nsep, samples)
    ref: (samples)
  Returns:
    chosen_sep: (samples)
  """
  ref = np.reshape(ref, [1, -1])
  idx = np.argmin(np.sum((np.abs(np.fft.rfft(sep)) - np.abs(np.fft.rfft(ref)))**2, axis=-1))
  return sep[idx]


def map_tracks_to_tracks(separated_tracks, reference_tracks, block_len=10000, max_sources_per_block=2, loss_fn=mag_mse_loss):
  """Map separated tracks to reference tracks using oracle mapping information.

  Args:
    separated_tracks: Separated tracks (N_sep, samples)
    reference_tracks: Reference tracks (N_max, samples)
    block_len: Block length for processing and mapping tracks.
    max_sources_per_block: Maximum number of sources per block.
  Returns:
    mapped_separated_tracks: Mapped separated tracks (N_max, samples) that
      matches with reference_tracks.
  """
  nsep, sep_samples = separated_tracks.shape
  nref, ref_samples = reference_tracks.shape
  assert sep_samples == ref_samples
  split_sep_tracks = np.split(separated_tracks, np.arange(block_len,sep_samples,block_len), axis=-1)
  split_ref_tracks = np.split(reference_tracks, np.arange(block_len,ref_samples,block_len), axis=-1)
  mapped_sep_tracks = []
  print(f'Num blocks {len(split_sep_tracks)}, shape {split_sep_tracks[0].shape}.')
  for sep_track, ref_track in zip(split_sep_tracks, split_ref_tracks):
    mapped_sep_track = np.zeros_like(ref_track)
    ref_en = np.sum(ref_track**2, axis=-1)
    top_idx = np.flip(np.argsort(ref_en))[:max_sources_per_block]
    for idx in list(top_idx):
      sep_match = loss_fn(sep_track, ref_track[idx])
      mapped_sep_track[idx] = sep_match
    mapped_sep_tracks.append(mapped_sep_track)
  mapped_sep_tracks_np = np.concatenate(mapped_sep_tracks, axis=-1)
  return mapped_sep_tracks_np
