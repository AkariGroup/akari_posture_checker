import blobconverter

blob_path = blobconverter.from_openvino(
    xml='models/openvino/human-pose-estimation-0001.xml',
    bin='models/openvino/human-pose-estimation-0001.bin',
    data_type='FP16',
    shaves=6,
    output_dir='models'
)

print(f"✅ Blob created at: {blob_path}")
