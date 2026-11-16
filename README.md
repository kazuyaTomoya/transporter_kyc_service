# XPDC transporter photo verification

FastAPI service used by the transporter creation form. It checks the KTP file against the selfie before the three document files are uploaded to permanent storage and before the transporter account is created.

## Checks

- Detects the KTP front and its NIK, name, portrait, and text regions inside the selfie. The detected regions must form a card-sized, front-facing group.
- Detects hand landmarks touching the card area.
- Compares the face detected on the KTP portrait with the largest non-card face in the selfie. The default cosine-similarity threshold is `0.60`, matching the linked repository's default.

The linked project only compares a KTP portrait with a selfie; it does not check that the KTP is held in the selfie. This service adds hand-landmark overlap with the detected card area and checks that the front-side portrait and text regions form a clear card-shaped group. These are image-based checks, not liveness detection, and should be calibrated using representative XPDC photos before production use.

## Run on the Hostinger VPS

1. Add `main.py`, `Dockerfile`, `requirements.txt`, `docker-compose.yaml`, and this README to the public GitHub repository. Do not add an API key or an `.env` file. The Dockerfile downloads the KTP detector checkpoint during build instead of storing the model binary in this repository.
2. In Hostinger Docker Manager, create a new Compose project named `transporter-kyc` from the `docker-compose.yaml` URL. Set `KYC_API_KEY` in the project Environment section to a private random value and set `KYC_FACE_THRESHOLD` to `0.60`.
3. The project creates its own Docker network and exposes port 8000 only inside that network. Connect `supabase-caddy` to this project network, then add a Caddy route for `/transporter-kyc/*` to `xpdc-transporter-kyc:8000`. Do not publish port 8000 to the internet.
4. Set these environment variables for the PHP application:

   ```text
   TRANSPORTER_KYC_API_URL=https://srv1606835.hstgr.cloud/transporter-kyc/verify
   TRANSPORTER_KYC_API_KEY=<same value as the project KYC_API_KEY>
   ```

   PHP must allow the two verification images in one request: set `upload_max_filesize` to at least `8M` and `post_max_size` to at least `16M`.

The service accepts multipart fields `ktp_image` and `selfie_image`, requires the `X-KYC-API-Key` header, limits each image to 7 MB, and does not write uploaded images to disk.

The face feature model downloads its pretrained FaceNet weights on first startup. The container therefore needs outbound internet access during its first initialization. CPU inference can take several seconds per request.

The linked GitHub repository does not currently include a license file. Confirm permission to use its KTP detector weights before deploying them beyond internal evaluation.
