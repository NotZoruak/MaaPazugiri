import path from 'node:path'

export default {
  cwd: import.meta.dirname,
  maaVersion: 'latest',
  maaCache: path.join(import.meta.dirname, '.cache', 'maa-tools'),
  interfacePath: 'interface.json',
  check: {},
  vscode: {
    agents: {
      uv: 'Maa Agent: Debug',
    },
  },
}
