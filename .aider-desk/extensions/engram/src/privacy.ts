/**
 * Secret redaction.
 *
 * Applied to conversation text BEFORE it is sent to the secondary LLM and
 * BEFORE any memory content is stored, so secrets never leave the machine to
 * the local endpoint's logs and never enter the vector store.
 *
 * This is a deliberately conservative, high-recall filter. It replaces matches
 * with [REDACTED] and never logs the matched text.
 */

interface Rule {
  name: string;
  re: RegExp;
}

const RULES: Rule[] = [
  // OpenAI / OpenAI-style keys
  { name: 'openai-key', re: /\bsk-[A-Za-z0-9_\-]{8,}\b/g },
  // Anthropic
  { name: 'anthropic-key', re: /\bsk-ant-[A-Za-z0-9_\-]{8,}\b/g },
  // GitHub tokens
  { name: 'github-token', re: /\bgh[pousr]_[A-Za-z0-9]{16,}\b/g },
  { name: 'github-ghs', re: /\bghs_[A-Za-z0-9]{16,}\b/g },
  // AWS access key ids
  { name: 'aws-key', re: /\b(AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b/g },
  // Slack
  { name: 'slack-token', re: /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/g },
  // Stripe
  { name: 'stripe-key', re: /\b[sr]k_(live|test)_[A-Za-z0-9]{16,}\b/g },
  // Google API keys
  { name: 'google-key', re: /\bAIza[0-9A-Za-z_\-]{30,}\b/g },
  // npm
  { name: 'npm-token', re: /\bnpm_[A-Za-z0-9]{30,}\b/g },
  // SendGrid / Twilio-ish prefixes
  { name: 'sendgrid-key', re: /\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b/g },
  { name: 'twilio-key', re: /\bSK[0-9a-fA-F]{32}\b/g },
  // Bearer / Basic auth headers
  { name: 'bearer', re: /\b(Bearer|Basic)\s+[A-Za-z0-9._\-]{12,}=*/gi },
  // x-api-key / api-key / authorization header style assignments
  { name: 'api-key-assign', re: /\b(x-api-key|api[-_]?key|access[-_]?token|auth[-_]?token|refresh[-_]?token|session[-_]?token|secret)\b\s*[:=]\s*["']?[A-Za-z0-9._\-]{8,}["']?/gi },
  // password / passwd / pwd assignments
  { name: 'password-assign', re: /\b(password|passwd|pwd)\b\s*[:=]\s*["']?[^\s"']{4,}["']?/gi },
  // private key blocks
  { name: 'private-key', re: /-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----/g },
  // JWTs (three base64url segments)
  { name: 'jwt', re: /\beyJ[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}\b/g },
  // Long high-entropy hex/base64 blobs that look like tokens (>=40 chars)
  { name: 'entropy-blob', re: /\b[A-Za-z0-9+/]{40,}={0,2}\b/g },
];

export interface RedactionResult {
  text: string;
  hits: number;
  categories: string[];
}

/**
 * Redact secrets from a block of text. Returns the sanitized text plus a count
 * of redactions and which rule categories fired (names only, never values).
 */
export function redactSecrets(text: string, enabled: boolean): RedactionResult {
  if (!enabled || !text) return { text: text ?? '', hits: 0, categories: [] };

  let out = text;
  let hits = 0;
  const categories = new Set<string>();

  for (const rule of RULES) {
    // Reset lastIndex for global regexes.
    rule.re.lastIndex = 0;
    out = out.replace(rule.re, () => {
      hits += 1;
      categories.add(rule.name);
      return '[REDACTED]';
    });
  }

  return { text: out, hits, categories: [...categories] };
}

/**
 * True if a candidate memory string appears to contain a secret. Used to drop
 * such candidates entirely rather than store a redacted shell of them.
 */
export function looksSecret(text: string, enabled: boolean): boolean {
  if (!enabled || !text) return false;
  return redactSecrets(text, true).hits > 0;
}
