# XPDC transporter photo verification

FastAPI service used by the transporter creation form. It checks the KTP file against the selfie before the three document files are uploaded to permanent storage and before the transporter account is created.

## Checks

- Detects the KTP front and its NIK, name, portrait, and text regions inside the selfie. The detected regions must form a card-sized, front-facing group.
- Detects hand landmarks touching the card area.
- Compares the face detected on the KTP portrait with the largest non-card face in the selfie. The default cosine-similarity threshold is `0.60`, matching the linked repository's default.

The linked project only compares a KTP portrait with a selfie; it does not check that the KTP is held in the selfie. This service adds hand-landmark overlap with the detected card area and checks that the front-side portrait and text regions form a clear card-shaped group. These are image-based checks, not liveness detection, and should be calibrated using representative XPDC photos before production use.

## Run on the VPS

1. Copy this directory to the VPS and create `.env` from `.env.example`. Set `KYC_API_KEY` to a private random value; do not expose it to browser JavaScript.
2. Build and start the container:

   ```sh
   docker compose up -d --build
   ```

3. Confirm the service locally on the VPS at `http://127.0.0.1:8183/health`.
4. Set these environment variables for the PHP application:

   ```text
   TRANSPORTER_KYC_API_URL=http://127.0.0.1:8183/verify
   TRANSPORTER_KYC_API_KEY=<same value as KYC_API_KEY>
   ```

   PHP must allow the two verification images in one request: set `upload_max_filesize` to at least `8M` and `post_max_size` to at least `16M`.

If PHP runs in a different container or host, `127.0.0.1` will point to that PHP environment. Connect the containers through a private Docker network or use a private TLS endpoint reachable only by the PHP host, then set `TRANSPORTER_KYC_API_URL` accordingly. Do not expose port 8183 publicly.

The service accepts multipart fields `ktp_image` and `selfie_image`, requires the `X-KYC-API-Key` header, limits each image to 7 MB, and does not write uploaded images to disk.

The face feature model downloads its pretrained FaceNet weights on first startup. The container therefore needs outbound internet access during its first initialization. CPU inference can take several seconds per request.

The linked GitHub repository does not currently include a license file. Confirm permission to use its KTP detector weights before deploying them beyond internal evaluation.
