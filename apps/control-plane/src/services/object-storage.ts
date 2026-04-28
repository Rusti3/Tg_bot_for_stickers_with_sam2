import { Readable } from "node:stream";

import {
  GetObjectCommand,
  HeadBucketCommand,
  PutObjectCommand,
  S3Client,
} from "@aws-sdk/client-s3";

import { AppConfig } from "../config";

export class ObjectStorageService {
  private readonly client: S3Client;

  constructor(private readonly config: AppConfig) {
    this.client = new S3Client({
      region: config.objectStorageRegion,
      endpoint: `${config.objectStorageUseSsl ? "https" : "http"}://${config.objectStorageEndpoint}`,
      forcePathStyle: true,
      credentials: {
        accessKeyId: config.objectStorageAccessKey,
        secretAccessKey: config.objectStorageSecretKey,
      },
    });
  }

  async ensureBucket(): Promise<void> {
    await this.client.send(new HeadBucketCommand({ Bucket: this.config.objectStorageBucket }));
  }

  async uploadBuffer(
    objectKey: string,
    buffer: Buffer,
    contentType: string,
  ): Promise<string> {
    await this.client.send(
      new PutObjectCommand({
        Bucket: this.config.objectStorageBucket,
        Key: objectKey,
        Body: buffer,
        ContentType: contentType,
      }),
    );
    return objectKey;
  }

  async downloadBuffer(objectKey: string): Promise<Buffer> {
    const response = await this.client.send(
      new GetObjectCommand({
        Bucket: this.config.objectStorageBucket,
        Key: objectKey,
      }),
    );
    return streamToBuffer(response.Body as Readable);
  }
}

async function streamToBuffer(stream: Readable): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of stream) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}
