import { Pool } from "pg";

import { AppConfig } from "./config";

export function createPool(config: AppConfig) {
  return new Pool({
    connectionString: config.databaseUrl,
    max: 10,
  });
}
