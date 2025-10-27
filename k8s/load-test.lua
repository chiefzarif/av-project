-- Load testing script for wrk
-- Usage: wrk -t4 -c100 -d30s -s load-test.lua http://visionguard:8000/predict

-- Read test image
local image_file = io.open("zidane.jpg", "rb")
local image_data = image_file:read("*all")
image_file:close()

-- Generate multipart form boundary
local boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

-- Construct multipart body
local body = string.format(
  "--%s\r\n" ..
  "Content-Disposition: form-data; name=\"file\"; filename=\"test.jpg\"\r\n" ..
  "Content-Type: image/jpeg\r\n\r\n" ..
  "%s\r\n" ..
  "--%s--\r\n",
  boundary, image_data, boundary
)

-- Setup request
request = function()
  return wrk.format(
    "POST",
    "/predict",
    {
      ["Content-Type"] = "multipart/form-data; boundary=" .. boundary,
      ["Content-Length"] = tostring(#body)
    },
    body
  )
end

-- Track response statistics
responses = {
  success = 0,
  errors = 0,
  total_latency = 0
}

-- Process response
response = function(status, headers, body)
  if status == 200 then
    responses.success = responses.success + 1
  else
    responses.errors = responses.errors + 1
  end
end

-- Print results
done = function(summary, latency, requests)
  io.write("---------------------------------------\n")
  io.write("Load Test Results:\n")
  io.write("---------------------------------------\n")
  io.write(string.format("  Requests:      %d\n", summary.requests))
  io.write(string.format("  Duration:      %.2fs\n", summary.duration / 1000000))
  io.write(string.format("  Requests/sec:  %.2f\n", summary.requests / (summary.duration / 1000000)))
  io.write(string.format("  Success:       %d\n", responses.success))
  io.write(string.format("  Errors:        %d\n", responses.errors))
  io.write(string.format("  Avg Latency:   %.2fms\n", latency.mean / 1000))
  io.write(string.format("  Max Latency:   %.2fms\n", latency.max / 1000))
  io.write(string.format("  P50 Latency:   %.2fms\n", latency:percentile(50) / 1000))
  io.write(string.format("  P95 Latency:   %.2fms\n", latency:percentile(95) / 1000))
  io.write(string.format("  P99 Latency:   %.2fms\n", latency:percentile(99) / 1000))
  io.write("---------------------------------------\n")
end
