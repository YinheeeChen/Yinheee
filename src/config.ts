import type {
  LicenseConfig,
  NavBarConfig,
  ProfileConfig,
  SiteConfig,
} from './types/config'
import { LinkPreset } from './types/config'

export const siteConfig: SiteConfig = {
  title: 'Yinheee',
  subtitle: 'Hello World!',
  lang: 'en',         // 'en', 'zh_CN', 'zh_TW', 'ja', 'ko', 'es', 'th'
  themeColor: {
    hue: 225,         // Default hue for the theme color, from 0 to 360. e.g. red: 0, teal: 200, cyan: 250, pink: 345
    fixed: false,     // Hide the theme color picker for visitors
  },
  // banner: {
  //   enable: true,
  //   src: 'assets/images/banner.mp4',   // Relative to the /src directory. Relative to the /public directory if it starts with '/'
  //   position: 'center',      // Equivalent to object-position, only supports 'top', 'center', 'bottom'. 'center' by default
  //   credit: {
  //     enable: false,         // Display the credit text of the banner image
  //     text: '',              // Credit text to be displayed
  //     url: ''                // (Optional) URL link to the original artwork or artist's page
  //   }
  // },
  banner: {
    enable: true,
    src: 'https://img.picgo.net/2025/04/01/17bd966f1b9dbf85e.mp4', 
    position: 'center',
    autoplay: true,
    loop: true,
    muted: true,
    credit: {
      enable: false,
      text: '',
      url: ''
    }
  },
  toc: {
    enable: true,           // Display the table of contents on the right side of the post
    depth: 2                // Maximum heading depth to show in the table, from 1 to 3
  },
  favicon: [    // Leave this array empty to use the default favicon
    {
      src: '/favicon/icon.png',    // Path of the favicon, relative to the /public directory
      theme: 'light',              // (Optional) Either 'light' or 'dark', set only if you have different favicons for light and dark mode
      sizes: '32x32',              // (Optional) Size of the favicon, set only if you have favicons of different sizes
    }
  ]
}

export const navBarConfig: NavBarConfig = {
  links: [
    LinkPreset.Home,
    LinkPreset.Archive,
    LinkPreset.About,
    {
      name: 'Friends',
      url: '/friends/',     // Internal links should not include the base path, as it is automatically added
      external: false,                               // Show an external link icon and will open in a new tab
    },
    {
      name: 'GitHub',
      url: 'https://github.com/YinheeeChen',     // Internal links should not include the base path, as it is automatically added
      external: true,                               // Show an external link icon and will open in a new tab
    },
  ],
}

export const profileConfig: ProfileConfig = {
  avatar: 'assets/images/1.jpg',  // Relative to the /src directory. Relative to the /public directory if it starts with '/'
  name: 'Yinheee',
  bio: 'Undergraduate Student of CSU.',
  links: [
    // {
    //   name: 'Twitter',
    //   icon: 'fa6-brands:twitter',       // Visit https://icones.js.org/ for icon codes
    //                                     // You will need to install the corresponding icon set if it's not already included
    //                                     // `pnpm add @iconify-json/<icon-set-name>`
    //   url: 'https://twitter.com',
    // },
    {
      name: 'GitHub',
      icon: 'fa6-brands:github',
      url: 'https://github.com/YinheeeChen',
    },
    {
      name: 'QQ',
      icon: 'fa6-brands:qq',
      url: 'https://qm.qq.com/q/7Rv1PspRyo',
    },
    {
      name: 'Email',
      icon: 'fa6-solid:envelope',
      url: 'mailto:yinhechen@csu.edu.cn',
    },
  ],
}

export const licenseConfig: LicenseConfig = {
  enable: true,
  name: 'CC BY-NC-SA 4.0',
  url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
}
