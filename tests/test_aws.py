import boto3

s3 = boto3.client("s3")

print("Testing correct path with sharepoint-files/ prefix...")
response = s3.list_objects_v2(Bucket="smart-meter-data-sb", Prefix="sharepoint-files/Zip4/202308/", MaxKeys=5)

if "Contents" in response:
    print(f"✅ Found {len(response['Contents'])} files!")
    print("\nSample files:")
    for obj in response["Contents"][:5]:
        print(f"  - {obj['Key']}")
        print(f"    Size: {obj['Size']:,} bytes")
else:
    print("❌ No files found")
