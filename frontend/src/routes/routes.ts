export const ROUTES = {
  overview: '/',
  library: '/library',
  chat: '/chat',
  matrix: '/matrix',
  history: '/history',
  status: '/status',
} as const

export type RouteKey = keyof typeof ROUTES
