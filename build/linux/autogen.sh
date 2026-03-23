#!/usr/bin/env bash

autoreconf --force --install --symlink


# Older versions of autoreconf (and/or libtool?) fail to install these for us.
# Broken on Debian 11.11, working on Debian 12.13
if test ! -e config.guess; then
  ln -s config.guess.dist config.guess
fi

if test ! -e config.sub; then
  ln -s config.sub.dist config.sub
fi

if test ! -e install-sh; then
  ln -s install-sh.dist install-sh
fi

echo "Done."
