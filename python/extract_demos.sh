#!/bin/bash
# Extract training data from all q2dm1 MVD demos
# Output: combined_training_data.bin

Q2AI="/home/oky/q2ai/q2ai"
BASEDIR="/home/oky/q2ai"
OUTDIR="/home/oky/q2ai"
DEMODIR="/home/oky/.q2pro/baseq2/demos"

# Find all q2dm1 MVD demos
DEMOS=$(ls "$DEMODIR"/*q2dm1*.mvd2 2>/dev/null)

if [ -z "$DEMOS" ]; then
    echo "No q2dm1 MVD demos found!"
    exit 1
fi

echo "Found demos:"
echo "$DEMOS" | while read d; do echo "  $(basename "$d")"; done

# Make sure demos are symlinked into q2ai basedir
mkdir -p "$BASEDIR/baseq2/demos"
for demo in $DEMOS; do
    ln -sf "$demo" "$BASEDIR/baseq2/demos/" 2>/dev/null
done

total_frames=0
data_files=""
i=0

for demo in $DEMOS; do
    name=$(basename "$demo" .mvd2)
    outfile="$OUTDIR/demo_data_${i}.bin"

    echo ""
    echo "=== Extracting: $name ==="

    # Remove old output
    rm -f "$OUTDIR/demo_training_data.bin"

    # Run extraction (timeout 120s per demo)
    cd "$BASEDIR"
    timeout 120 "$Q2AI" +set q2ai_demo 1 +set logfile 0 +set basedir "$BASEDIR" \
        "+mvdplay $name" > /dev/null 2>&1

    if [ -f "$OUTDIR/demo_training_data.bin" ]; then
        size=$(stat -c%s "$OUTDIR/demo_training_data.bin")
        if [ "$size" -gt 100 ]; then
            mv "$OUTDIR/demo_training_data.bin" "$outfile"
            frames=$(python3 -c "
import struct, os
f = open('$outfile', 'rb')
obs_size, act_size = struct.unpack('ii', f.read(8))
n = (os.path.getsize('$outfile') - 8) // (obs_size + act_size)
print(n)
f.close()
")
            echo "  -> $frames frames"
            total_frames=$((total_frames + frames))
            data_files="$data_files $outfile"
        else
            echo "  -> Too small, skipping"
            rm -f "$outfile"
        fi
    else
        echo "  -> No output"
    fi

    i=$((i + 1))
done

echo ""
echo "=== Total: $total_frames frames from $(echo $data_files | wc -w) demos ==="

# Combine all data files
if [ -n "$data_files" ]; then
    echo "Combining data files..."
    python3 -c "
import struct, sys, os

files = '$data_files'.split()
all_obs = b''
all_act = b''
obs_size = act_size = 0
total = 0

for f in files:
    with open(f, 'rb') as fp:
        o, a = struct.unpack('ii', fp.read(8))
        if obs_size == 0:
            obs_size, act_size = o, a
        data = fp.read()
        pair_size = obs_size + act_size
        n = len(data) // pair_size
        all_obs += data[:n * pair_size]
        total += n

with open('$OUTDIR/combined_training_data.bin', 'wb') as fp:
    fp.write(struct.pack('ii', obs_size, act_size))
    fp.write(all_obs)

print(f'Combined {total} frames into combined_training_data.bin')
"
fi
