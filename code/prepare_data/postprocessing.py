import os

ASSET_PATH = "/root/asset"

def mask_zero():

    non_speeches = None
    with open(os.path.join(ASSET_PATH, "nonspeech.csv")) as f:
        f.readline()
        non_speeches = [non_speech.strip() for non_speech in f.readlines()]

    with open(os.path.join(ASSET_PATH, "submission.csv"), "r") as f, \
         open(os.path.join(ASSET_PATH, "masked_submission.csv"), "w") as wf:
        
        wf.write(f.readline())
        for line in f.readlines():
            _id, _fake, _real = line.strip().split(",")
            if _id in non_speeches:
                _fake, _real = 0., 0.
            wf.write("{},{},{}\n".format(_id, _fake, _real))
        wf.close()

    print("Masking (post-processing) done!")


if __name__ == "__main__":
    mask_zero()