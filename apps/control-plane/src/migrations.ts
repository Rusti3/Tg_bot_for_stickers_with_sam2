import fs from "node:fs/promises";
import path from "node:path";

interface DbLike {
  query<T = any>(text: string, params?: any[]): Promise<{ rows: T[]; rowCount: number | null }>;
  connect(): Promise<{
    query<T = any>(text: string, params?: any[]): Promise<{ rows: T[]; rowCount: number | null }>;
    release(): void;
  }>;
}

export async function runMigrations(pool: DbLike): Promise<void> {
  const migrationsDir = path.resolve(__dirname, "..", "..", "migrations");
  const files = (await fs.readdir(migrationsDir))
    .filter((name) => name.endsWith(".sql"))
    .sort((left, right) => left.localeCompare(right));

  await pool.query(`
    create table if not exists schema_migrations (
      name text primary key,
      applied_at timestamptz not null default now()
    )
  `);

  for (const file of files) {
    const alreadyApplied = await pool.query<{ name: string }>(
      "select name from schema_migrations where name = $1",
      [file],
    );
    if (alreadyApplied.rowCount) {
      continue;
    }

    const sql = await fs.readFile(path.join(migrationsDir, file), "utf8");
    const client = await pool.connect();
    try {
      await client.query("begin");
      await client.query(sql);
      await client.query("insert into schema_migrations(name) values ($1)", [file]);
      await client.query("commit");
    } catch (error) {
      await client.query("rollback");
      throw error;
    } finally {
      client.release();
    }
  }
}
